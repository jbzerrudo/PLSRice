import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50  # Increase the warning threshold

import os
import traceback
import numpy as np
import sympy as sp
import pandas as pd
from itertools import combinations
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold


def calculate_explained_variance(X, pls_model):
    """Calculate explained variance ratio for a PLS model manually."""
    # Get the original variance
    X_std = np.std(X, axis=0, ddof=1)
    total_var = np.sum(X_std ** 2)
    
    # Get the X loadings
    loadings = pls_model.x_loadings_
    
    # Calculate variance explained by each component
    explained_var = []
    for i in range(loadings.shape[1]):
        comp_loading = loadings[:, i]
        comp_var = np.sum(comp_loading ** 2)
        explained_var.append(comp_var / total_var)
    
    return np.array(explained_var)

def analyze_variable_importance(X, Y, n_components=2):
    """Comprehensive analysis of variable importance in PLS regression"""
    try:
        # Ensure input is numpy array
        X = X.values if hasattr(X, 'values') else np.asarray(X)
        Y = Y.values if hasattr(Y, 'values') else np.asarray(Y)
        
        # Scale the data
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        Y_scaled = Y_scaler.fit_transform(Y)
        
        # Fit PLS model
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_scaled, Y_scaled)
        
        # Calculate correlation loadings for X variables
        x_corr_loadings = np.zeros((X.shape[1], n_components))
        for i in range(X.shape[1]):
            for j in range(n_components):
                x_corr_loadings[i, j] = np.corrcoef(X_scaled[:, i], pls.x_scores_[:, j])[0, 1]
        
        # Calculate absolute correlation loadings (importance metric)
        abs_corr_loadings = np.abs(x_corr_loadings)
        
        # Calculate variable importance based on correlation loadings
        importance_by_loadings = np.sqrt(np.sum(abs_corr_loadings**2, axis=1))
        
        # Calculate VIP scores
        vip = vip_scores(pls, X_scaled)
        
        # Rank variables by importance
        var_importance_ranking = np.argsort(vip)[::-1]
        
        # Calculate explained variance manually
        variance_explained = calculate_explained_variance(X_scaled, pls)
        
        # Prepare results dictionary
        results = {
            'correlation_loadings': x_corr_loadings,
            'abs_correlation_loadings': abs_corr_loadings,
            'importance_scores': importance_by_loadings,
            'vip_scores': vip,
            'importance_ranking': var_importance_ranking,
            'variance_explained': variance_explained,
            'pls_model': pls,
            'X_scaler': X_scaler,
            'Y_scaler': Y_scaler
        }
        
        return results
    except Exception as e:
        print(f"Error in analyze_variable_importance: {e}")
        traceback.print_exc()  # Print full traceback
        return None
    
def vip_scores(model, X):
    """Calculate VIP scores for PLS model"""
    try:
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        
        p = w.shape[0]
        vips = np.zeros(p)
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(w.shape[1])])
            # Fixed calculation to avoid NumPy warning
            vips[i] = np.sqrt(p * np.sum(weight.reshape(-1, 1) * s) / total_s)
        
        return vips
        
    except Exception as e:
        print(f"Error in VIP calculation: {e}")
        # Return default values based on regression coefficients
        return np.abs(model.coef_.ravel())

def find_optimal_components(X, y, max_components=10):
    """Find optimal number of components using cross-validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Ensure y is numpy array
    y_array = y.values if hasattr(y, 'values') else y
    
    # Scale the data
    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y_array.reshape(-1, 1)).ravel()
    
    # Test different numbers of components
    component_numbers = range(1, min(max_components + 1, X.shape[1] + 1))
    cv_scores = []
    
    for n_comp in component_numbers:
        pls = PLSRegression(n_components=n_comp)
        scores = cross_val_score(pls, X_scaled, y_scaled, cv=kf, scoring='neg_mean_squared_error')
        cv_scores.append(-np.mean(scores))  # Convert to positive MSE
    
    # Plot cross-validation results - increase figure size
    plt.figure(figsize=(12, 8))  # Larger figure size
    plt.plot(component_numbers, cv_scores, 'o-')
    plt.xlabel('Number of PLS Components')
    plt.ylabel('Mean Squared Error (Cross-Validation)')
    plt.title('PLS Model Optimization: Selecting Number of Components')
    plt.grid(True, alpha=0.3)
    
    # Find optimal number of components (lowest MSE)
    optimal_components = component_numbers[np.argmin(cv_scores)]
    plt.axvline(x=optimal_components, color='red', linestyle='--')
    
    # Adjust text position and add padding
    text_x = optimal_components + 0.2
    text_y = min(cv_scores) * 1.1
    plt.text(text_x, text_y, f'Optimal: {optimal_components} components', 
             color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add padding to figure edges
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
    
    # Return without tight_layout to avoid warnings
    # plt.tight_layout(pad=1.5)  # Remove this line
    
    return optimal_components, cv_scores

def select_optimal_variables(X, y, max_features=None, step=1, importance_results=None):
    """
    Select optimal number of variables based on recursive feature elimination.
    
    Parameters:
    -----------
    X : DataFrame or array
        Predictor variables
    y : Series or array
        Response variable
    max_features : int, optional
        Maximum number of features to consider
    step : int, optional
        Step size for feature elimination
    importance_results : dict, optional
        Results from analyze_variable_importance function to prioritize variable selection
        
    Returns:
    --------
    tuple : (list of optimal feature indices, optimal number of features, cross-validation scores)
    """
    # If importance_results provided, use it to pre-rank variables
    if importance_results is not None:
        # Get the ranking from VIP scores
        ranking = importance_results['importance_ranking']
        X_cols = X.columns if hasattr(X, 'columns') else None
        
        if X_cols is not None:
            print("\nPre-ranked variables by importance:")
            for i, idx in enumerate(ranking[:20]):
                print(f"{i+1}. {X_cols[idx]} (VIP: {importance_results['vip_scores'][idx]:.4f})")
    else:
        # Use linear model for recursive feature elimination
        estimator = LinearRegression()
        
        # Set up RFE
        if max_features is None:
            max_features = X.shape[1] // 2
        
        rfe = RFE(estimator, n_features_to_select=1, step=step)
        rfe.fit(X, y)
        ranking = np.argsort(rfe.ranking_)
    
    # Test different feature subsets using cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define feature subset sizes to test
    if max_features is None:
        max_features = len(ranking)
    
    feature_numbers = range(1, min(max_features + 1, len(ranking) + 1))
    cv_scores = []
    r2_scores = []
    
    for n_features in feature_numbers:
        # Select top n_features from ranking
        selected_features = ranking[:n_features]
        X_subset = X.iloc[:, selected_features] if hasattr(X, 'iloc') else X[:, selected_features]
        
        # Ensure y is numpy array for find_optimal_components
        y_numpy = y.values if hasattr(y, 'values') else y
        
        # Find optimal components for this subset
        opt_components, _ = find_optimal_components(X_subset, y_numpy, max_components=min(5, n_features))
        
        # Evaluate with cross-validation
        pls = PLSRegression(n_components=opt_components)
        mse_scores = cross_val_score(pls, X_subset, y, cv=kf, 
                                    scoring='neg_mean_squared_error')
        r2_scores_cv = cross_val_score(pls, X_subset, y, cv=kf, 
                                      scoring='r2')
        
        cv_scores.append(-np.mean(mse_scores))  # Convert to positive MSE
        r2_scores.append(np.mean(r2_scores_cv))
    
    # Plot cross-validation results - increase figure size
    fig, ax1 = plt.subplots(figsize=(14, 9))  # Larger figure size
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', color=color, fontsize=12)
    ax1.plot(feature_numbers, cv_scores, 'o-', color=color, markersize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('R² Score', color=color, fontsize=12)
    ax2.plot(feature_numbers, r2_scores, 's-', color=color, markersize=5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Find optimal number of features (could be based on MSE or R²)
    # Let's use the elbow point detection for MSE
    # Simple approach: find where the improvement becomes marginal
    mse_diffs = np.diff(cv_scores)
    # Add a small number to prevent division by zero
    mse_improvements = np.abs(mse_diffs / (np.array(cv_scores[1:]) + 1e-10))
    
    # Find where improvement is less than 1%
    small_improvements = np.where(mse_improvements < 0.01)[0]
    if len(small_improvements) > 0:
        optimal_n_features = feature_numbers[small_improvements[0] + 1]
    else:
        # If no clear elbow, pick the best R² score that's not overfitting
        r2_diffs = np.diff(r2_scores)
        r2_improvements = np.abs(r2_diffs / (np.array(r2_scores[1:]) + 1e-10))
        small_r2_improvements = np.where(r2_improvements < 0.01)[0]
        
        if len(small_r2_improvements) > 0:
            optimal_n_features = feature_numbers[small_r2_improvements[0] + 1]
        else:
            # If still no clear point, use the number of features at maximum R²
            optimal_n_features = feature_numbers[np.argmax(r2_scores)]
    
    # Mark the optimal number of features
    ax1.axvline(x=optimal_n_features, color='green', linestyle='--', linewidth=2)
    plt.title(f'Feature Selection: Optimal number of features = {optimal_n_features}', fontsize=14)
    
    # Add legend with better positioning
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(['MSE Score', 'Optimal Features'], loc='upper right', frameon=True)
    ax2.legend(['R² Score'], loc='upper left', frameon=True)
    
    # Add padding to figure edges
    plt.subplots_adjust(left=0.12, right=0.90, top=0.92, bottom=0.12)
    
    # Return optimal feature indices, number of features, and scores
    optimal_features = ranking[:optimal_n_features]
    
    return optimal_features, optimal_n_features, (cv_scores, r2_scores)

def generate_empirical_equation(X, y, selected_features, n_components=None, variable_names=None):
    """
    Generate an empirical equation from PLS regression model with improved precision.
    """
    # Extract selected features
    X_selected = X.iloc[:, selected_features] if hasattr(X, 'iloc') else X[:, selected_features]
    
    # If variable names not provided, create generic ones
    if variable_names is None:
        if hasattr(X, 'columns'):
            variable_names = X.columns[selected_features].tolist()
        else:
            variable_names = [f'X{i+1}' for i in range(len(selected_features))]
    else:
        if len(variable_names) == X.shape[1]:
            variable_names = [variable_names[i] for i in selected_features]
        elif len(variable_names) == len(selected_features):
            pass
        else:
            if hasattr(X, 'columns'):
                variable_names = X.columns[selected_features].tolist()
            else:
                variable_names = [f'X{i+1}' for i in range(len(selected_features))]
    
    # Ensure y is numpy array
    y_array = y.values if hasattr(y, 'values') else y
    
    # Determine optimal number of components if not provided
    if n_components is None:
        n_components, _ = find_optimal_components(X_selected, y_array, max_components=min(10, len(selected_features)))
    else:
        n_components = min(n_components, len(selected_features))
    
    # Fit PLS model
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_selected)
    y_scaled = scaler_y.fit_transform(y_array.reshape(-1, 1))
    
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_scaled, y_scaled)
    
    # Extract regression coefficients
    coefs = pls.coef_.flatten()
    
    # **NEW: Higher precision coefficient formatting**
    def format_coefficient(coef, precision=4):
        """Format coefficients with appropriate significant figures"""
        if abs(coef) >= 1:
            return f"{coef:.{precision}f}"
        elif abs(coef) >= 0.001:
            return f"{coef:.{precision+2}f}"
        else:
            return f"{coef:.2e}"  # Scientific notation for very small values
    
    # Calculate the intercept (in the original scale) with higher precision
    try:
        from decimal import Decimal, getcontext
        getcontext().prec = 28  # **NEW: High precision for coefficients**
        
        scaler_y_mean = float(scaler_y.mean_[0] if hasattr(scaler_y.mean_, 'shape') else scaler_y.mean_)
        scaler_y_scale = float(scaler_y.scale_[0] if hasattr(scaler_y.scale_, 'shape') else scaler_y.scale_)
        
        sum_term = Decimal('0')
        for i in range(len(coefs)):
            coef_decimal = Decimal(str(coefs[i]))
            mean_decimal = Decimal(str(scaler_X.mean_[i]))
            scale_decimal = Decimal(str(scaler_X.scale_[i]))
            sum_term += coef_decimal * mean_decimal / scale_decimal
        
        intercept = float(Decimal(str(scaler_y_mean)) - sum_term * Decimal(str(scaler_y_scale)))
        
    except Exception as e:
        print(f"High precision calculation failed: {e}, using standard precision")
        scaler_y_mean = scaler_y.mean_[0] if hasattr(scaler_y.mean_, 'shape') else scaler_y.mean_
        scaler_y_scale = scaler_y.scale_[0] if hasattr(scaler_y.scale_, 'shape') else scaler_y.scale_
        
        sum_term = 0
        for i in range(len(coefs)):
            sum_term += coefs[i] * scaler_X.mean_[i] / scaler_X.scale_[i]
        
        intercept = float(scaler_y_mean - sum_term * scaler_y_scale)
    
    # Store raw coefficient values with higher precision
    raw_coef_values = []
    var_names_with_coefs = []
    
    for i in range(len(coefs)):
        try:
            scaler_y_scale = scaler_y.scale_[0] if hasattr(scaler_y.scale_, 'shape') else scaler_y.scale_
            x_scale = scaler_X.scale_[i]
            scaled_coef = float(coefs[i] * scaler_y_scale / x_scale)
            raw_coef_values.append(scaled_coef)
            var_names_with_coefs.append(variable_names[i])
        except Exception as e:
            print(f"Warning in coefficient calculation for term {i}: {e}")
            raw_coef_values.append(0.0)
            var_names_with_coefs.append(variable_names[i])
    
    # **NEW: Format equation with improved precision**
    eq_readable = ""
    first_term = True
    
    # Add intercept first if significant
    if abs(intercept) >= 0.0001:
        eq_readable = format_coefficient(intercept)
        first_term = False
    
    for i, (name, coef) in enumerate(zip(var_names_with_coefs, raw_coef_values)):
        if abs(coef) < 0.0001:  # Skip very small coefficients
            continue
            
        if first_term:
            if coef < 0:
                eq_readable = f"-{format_coefficient(abs(coef))} * {name}"
            else:
                eq_readable = f"{format_coefficient(coef)} * {name}"
            first_term = False
        else:
            if coef < 0:
                eq_readable += f" - {format_coefficient(abs(coef))} * {name}"
            else:
                eq_readable += f" + {format_coefficient(coef)} * {name}"
    
    if first_term:  # Only intercept
        eq_readable = format_coefficient(intercept)
    
    # Calculate model evaluation metrics
    y_pred = pls.predict(X_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred).flatten()
    y_flat = y_array.flatten() if hasattr(y_array, 'flatten') else y_array
    
    mse = np.mean((y_flat - y_pred_original) ** 2)
    r2 = 1 - np.sum((y_flat - y_pred_original) ** 2) / np.sum((y_flat - np.mean(y_flat)) ** 2)
    
    equation_info = {
        'equation_string': eq_readable,
        'equation_readable': eq_readable,
        'symbolic_equation': None,  # Can add sympy if needed
        'pls_model': pls,
        'selected_features': selected_features,
        'variable_names': variable_names,
        'raw_coefficients': dict(zip(var_names_with_coefs, raw_coef_values)),
        'intercept': intercept,
        'X_scaler': scaler_X,
        'y_scaler': scaler_y,
        'mse': mse,
        'r2': r2,
        'n_components': n_components
    }
    
    return equation_info

def plot_observed_vs_predicted(y_true, equation_info, X=None, title=None):
    """Plot observed vs predicted values using the PLS equation"""
    # Extract model components
    pls = equation_info['pls_model']
    selected_features = equation_info['selected_features']
    X_scaler = equation_info['X_scaler']
    y_scaler = equation_info['y_scaler']
    
    # Convert y_true to numpy if it's a Series
    y_true_array = y_true.values if hasattr(y_true, 'values') else y_true
    
    # Make predictions
    X_selected = X.iloc[:, selected_features] if hasattr(X, 'iloc') else X[:, selected_features]
    X_scaled = X_scaler.transform(X_selected)
    y_pred_scaled = pls.predict(X_scaled)
    
    # Ensure prediction has correct shape for inverse transform (must be 2D)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Ensure y_true is flattened for plotting
    y_true_flat = y_true_array.flatten() if hasattr(y_true_array, 'flatten') else y_true_array
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred, alpha=0.7)
    
    # Add diagonal line (perfect prediction)
    min_val = min(np.min(y_true_flat), np.min(y_pred))
    max_val = max(np.max(y_true_flat), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    
    if title is None:
        plt.title(f'Observed vs Predicted (R² = {equation_info["r2"]:.4f})')
    else:
        plt.title(title)
    
    # Add R² and MSE to the plot
    plt.text(0.05, 0.95, f'R² = {equation_info["r2"]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'MSE = {equation_info["mse"]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'Components = {equation_info["n_components"]}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, f'# Variables = {len(selected_features)}', transform=plt.gca().transAxes)
    
    plt.grid(alpha=0.3)
    plt.tight_layout(pad=1.5)
    
    return plt.gcf()

def create_correlation_loadings_plot(X, Y, n_components=2, plot_components=(0, 1), 
                                     x_labels=None, y_labels=None, 
                                     title="PLS Correlation Loadings Plot",
                                     hotelling_t2=True,
                                     confidence_level=0.95):
    """
    Create a correlation loadings plot similar to The Unscrambler X for PLS regression.
    
    Parameters:
    -----------
    X : numpy array or pandas DataFrame
        X data matrix (predictors)
    Y : numpy array or pandas DataFrame
        Y data matrix (responses)
    n_components : int
        Number of components to use in PLS model
    plot_components : tuple
        Which components to plot (0-indexed)
    x_labels : list
        Labels for X variables
    y_labels : list
        Labels for Y variables
    title : str
        Plot title
    hotelling_t2 : bool
        Whether to plot Hotelling's T² ellipse
    confidence_level : float
        Confidence level for Hotelling's T² ellipse (0.95 = 95%)
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure object
    pls : PLSRegression model
        The fitted PLS model
    """
    # Ensure we have numpy arrays
    if isinstance(X, pd.DataFrame):
        if x_labels is None:
            x_labels = X.columns.tolist()
        X = X.values
    else:
        if x_labels is None:
            x_labels = [f"X{i+1}" for i in range(X.shape[1])]
            
    if isinstance(Y, pd.DataFrame):
        if y_labels is None:
            y_labels = Y.columns.tolist()
        Y = Y.values
    else:
        if y_labels is None:
            y_labels = [f"Y{i+1}" for i in range(Y.shape[1])]
    
    # Scale data
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)
    
    # Fit PLS model
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_scaled, Y_scaled)
    
    # Get scores (T and U)
    T = pls.x_scores_
    U = pls.y_scores_
    
    # Calculate correlation loadings for X variables
    x_corr_loadings = np.zeros((X.shape[1], n_components))
    for i in range(X.shape[1]):
        for j in range(n_components):
            x_corr_loadings[i, j] = np.corrcoef(X_scaled[:, i], T[:, j])[0, 1]
    
    # Calculate correlation loadings for Y variables
    y_corr_loadings = np.zeros((Y.shape[1], n_components))
    for i in range(Y.shape[1]):
        for j in range(n_components):
            y_corr_loadings[i, j] = np.corrcoef(Y_scaled[:, i], T[:, j])[0, 1]
    
    # Calculate explained variance manually
    expl_var_x = calculate_explained_variance(X_scaled, pls)
    
    # Create the plot with increased size
    fig, ax = plt.subplots(figsize=(14, 12))  # Larger figure size
    
    # Plot inner and outer circles (50% and 100% explained variance)
    circle1 = plt.Circle((0, 0), 0.5, color='gray', fill=False, linestyle='--', alpha=0.5)
    circle2 = plt.Circle((0, 0), 1.0, color='gray', fill=False, alpha=0.5)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    
    # Extract the components to plot
    pc1, pc2 = plot_components
    
    # Calculate Hotelling's T² ellipse (if requested)
    if hotelling_t2:
        # Extract scores for the selected components
        scores = T[:, [pc1, pc2]]
        
        # Calculate the covariance matrix of the scores
        cov_matrix = np.cov(scores, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate the critical value from F-distribution
        from scipy import stats
        n = X.shape[0]
        p = 2  # Two components
        critical_value = p * (n-1) / (n-p) * stats.f.ppf(confidence_level, p, n-p)
        
        # Calculate the radii of the ellipse
        radii = np.sqrt(eigenvalues * critical_value)
        
        # Create points for the ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = radii[0] * np.cos(theta)
        ellipse_y = radii[1] * np.sin(theta)
        
        # Rotate the ellipse
        R = eigenvectors
        ellipse_points = np.dot(np.column_stack([ellipse_x, ellipse_y]), R)
        
        # Plot the ellipse
        ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', lw=2, 
                label=f"Hotelling's T² ({confidence_level:.0%})")
    
    # Plot X-variable correlation loadings
    for i, label in enumerate(x_labels):
        ax.plot(x_corr_loadings[i, pc1], x_corr_loadings[i, pc2], 'bo', label='_nolegend_')
        # Use text instead of annotate for better control
        ax.text(x_corr_loadings[i, pc1] + 0.02, x_corr_loadings[i, pc2] + 0.02, label,
                color='blue', fontsize=9, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
    
    # Plot Y-variable correlation loadings
    for i, label in enumerate(y_labels):
        ax.plot(y_corr_loadings[i, pc1], y_corr_loadings[i, pc2], 'ro', label='_nolegend_')
        ax.text(y_corr_loadings[i, pc1] + 0.02, y_corr_loadings[i, pc2] + 0.02, label,
                color='red', fontsize=9, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
    
    # Add legend for X and Y variables and Hotelling's T² ellipse
    ax.plot([], [], 'bo', label='X-variables')
    ax.plot([], [], 'ro', label='Y-variables')
    ax.legend(loc='best', frameon=True, fontsize=10)
    
    # Add gridlines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set axis limits slightly beyond the unit circle
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Label axes with explained variance
    ax.set_xlabel(f"PC{pc1+1} ({expl_var_x[pc1]:.1%} explained X-variance)", fontsize=12)
    ax.set_ylabel(f"PC{pc2+1} ({expl_var_x[pc2]:.1%} explained X-variance)", fontsize=12)
    
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    # Add padding to figure edges
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    return fig, pls

def compare_variable_sets(X, y, importance_results, max_vars=20, n_components=None):
    """
    Compare model performance with increasing numbers of variables based on VIP scores.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable (CH4)
    importance_results : dict
        Results from analyze_variable_importance function with VIP scores
    max_vars : int
        Maximum number of variables to test
    n_components : int or None
        Number of PLS components to use (if None, optimal will be determined)
        
    Returns:
    --------
    DataFrame with performance metrics for each model
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score, KFold
    
    # Get variable ranking from VIP scores
    ranking = importance_results['importance_ranking']
    
    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = []
    
    # Test with increasing numbers of top variables
    for n_vars in range(1, min(max_vars + 1, len(ranking) + 1)):
        # Select top n_vars from ranking
        selected_features = ranking[:n_vars]
        X_subset = X.iloc[:, selected_features]
        
        # Determine optimal components if not provided
        if n_components is None:
            n_comp, _ = find_optimal_components(X_subset, y, max_components=min(10, n_vars))
        else:
            n_comp = min(n_components, n_vars)  # Can't have more components than variables
        
        # Scale data
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X_subset)
        
        # Create and fit PLS model
        pls = PLSRegression(n_components=n_comp, scale=False)
        
        # Cross-validation for R²
        r2_scores = cross_val_score(pls, X_scaled, y, cv=kf, scoring='r2')
        mean_r2 = np.mean(r2_scores)
        
        # Cross-validation for MSE
        mse_scores = -cross_val_score(pls, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        mean_mse = np.mean(mse_scores)
        
        # Fit on full dataset to get variable information
        pls.fit(X_scaled, y)
        
        # Get variable names
        var_names = X.columns[selected_features].tolist()
        
        # Store results
        results.append({
            'n_variables': n_vars,
            'variables': var_names,
            'n_components': n_comp,
            'r2_cv': mean_r2,
            'mse_cv': mean_mse
        })
    
    return pd.DataFrame(results)

def plot_model_comparison(results_df):
    """
    Plot R² and MSE for models with different numbers of variables.
    
    Parameters:
    -----------
    results_df : DataFrame
        Output from compare_variable_sets function
    """
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot R² on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of Variables', fontsize=12)
    ax1.set_ylabel('R² Score (Cross-Validated)', color=color, fontsize=12)
    ax1.plot(results_df['n_variables'], results_df['r2_cv'], 'o-', color=color, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot MSE on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean Squared Error', color=color, fontsize=12)
    ax2.plot(results_df['n_variables'], results_df['mse_cv'], 's-', color=color, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add details for each point
    for i, row in results_df.iterrows():
        ax1.annotate(f"{row['r2_cv']:.4f}", 
                   xy=(row['n_variables'], row['r2_cv']),
                   xytext=(5, 5), textcoords='offset points', color='red', fontsize=9)
        ax2.annotate(f"{row['mse_cv']:.4f}", 
                   xy=(row['n_variables'], row['mse_cv']),
                   xytext=(5, -15), textcoords='offset points', color='blue', fontsize=9)
    
    # Find best models based on R² and MSE
    best_r2_idx = results_df['r2_cv'].idxmax()
    best_mse_idx = results_df['mse_cv'].idxmin()
    
    # Mark best R²
    ax1.axvline(x=results_df.loc[best_r2_idx, 'n_variables'], color='red', linestyle='--', alpha=0.5)
    ax1.annotate(f"Best R² = {results_df.loc[best_r2_idx, 'r2_cv']:.4f}\n{results_df.loc[best_r2_idx, 'n_variables']} variables",
               xy=(results_df.loc[best_r2_idx, 'n_variables'], results_df.loc[best_r2_idx, 'r2_cv']),
               xytext=(10, 20), textcoords='offset points', color='red', fontsize=10,
               arrowprops=dict(arrowstyle="->", color='red'))
    
    # Mark best MSE
    ax2.axvline(x=results_df.loc[best_mse_idx, 'n_variables'], color='blue', linestyle='--', alpha=0.5)
    ax2.annotate(f"Best MSE = {results_df.loc[best_mse_idx, 'mse_cv']:.4f}\n{results_df.loc[best_mse_idx, 'n_variables']} variables",
               xy=(results_df.loc[best_mse_idx, 'n_variables'], results_df.loc[best_mse_idx, 'mse_cv']),
               xytext=(-100, -30), textcoords='offset points', color='blue', fontsize=10,
               arrowprops=dict(arrowstyle="->", color='blue'))
    
    plt.title('Model Performance vs. Number of Variables (Cross-Validated)', fontsize=14)
    plt.tight_layout()
    
    return fig

def generate_model_summary(results_df, X, importance_results):
    """Generate text summary of results with variable details"""
    
    best_r2_idx = results_df['r2_cv'].idxmax()
    best_mse_idx = results_df['mse_cv'].idxmin()
    
    # Get variable details for best models
    best_r2_vars = results_df.loc[best_r2_idx, 'variables']
    best_mse_vars = results_df.loc[best_mse_idx, 'variables']
    
    # Get variable importance details
    var_rankings = importance_results['importance_ranking']
    vip_scores = importance_results['vip_scores']
    
    # Create summary text
    summary = "MODEL COMPARISON SUMMARY\n"
    summary += "=" * 50 + "\n\n"
    
    # Check if there's a model with 5 variables
    if any(results_df['n_variables'] == 5):
        current_model = results_df.loc[results_df['n_variables'] == 5]
        summary += f"CURRENT MODEL (5 variables):\n"
        summary += f"R² = {current_model['r2_cv'].values[0]:.4f}\n"
        summary += f"MSE = {current_model['mse_cv'].values[0]:.4f}\n"
        summary += f"Variables: {', '.join(current_model['variables'].values[0])}\n\n"
        r2_improvement = results_df.loc[best_r2_idx, 'r2_cv'] - current_model['r2_cv'].values[0]
        
        additional_vars = [var for var in best_r2_vars if var not in current_model['variables'].values[0]]
    else:
        summary += "CURRENT MODEL (5 variables): Not found in results\n\n"
        r2_improvement = 0
        additional_vars = best_r2_vars
    
    summary += f"BEST MODEL BY R² ({results_df.loc[best_r2_idx, 'n_variables']} variables):\n"
    summary += f"R² = {results_df.loc[best_r2_idx, 'r2_cv']:.4f} (improvement: {r2_improvement:.4f})\n"
    summary += f"MSE = {results_df.loc[best_r2_idx, 'mse_cv']:.4f}\n"
    summary += f"Components: {results_df.loc[best_r2_idx, 'n_components']}\n"
    summary += "Variables:\n"
    
    # List variables in the best R² model with their importance scores
    for i, var in enumerate(best_r2_vars):
        var_idx = list(X.columns).index(var)
        var_rank = list(var_rankings).index(var_idx) + 1 if var_idx in var_rankings else "N/A"
        var_vip = vip_scores[var_idx] if var_idx < len(vip_scores) else "N/A"
        summary += f"{i+1}. {var} (Rank: {var_rank}, VIP: {var_vip:.4f})\n"
    
    summary += "\nADDITIONAL VARIABLES COMPARED TO CURRENT MODEL:\n"
    if additional_vars:
        for var in additional_vars:
            var_idx = list(X.columns).index(var)
            var_rank = list(var_rankings).index(var_idx) + 1 if var_idx in var_rankings else "N/A"
            var_vip = vip_scores[var_idx] if var_idx < len(vip_scores) else "N/A"
            summary += f"+ {var} (Rank: {var_rank}, VIP: {var_vip:.4f})\n"
    else:
        summary += "None\n"
    
    return summary

def build_model_with_target_mse(X, y, importance_results, target_mse=0.0090, max_vars=20):
    """
    Build a model that achieves a target MSE by incrementally adding variables.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable (CH4)
    importance_results : dict
        Results from analyze_variable_importance function with VIP scores
    target_mse : float
        Target Mean Squared Error to achieve
    max_vars : int
        Maximum number of variables to consider
        
    Returns:
    --------
    Dictionary with model details
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    
    # Get variable ranking from VIP scores
    ranking = importance_results['importance_ranking']
    
    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize variables
    best_n_vars = 0
    best_mse = float('inf')
    best_r2 = 0
    best_components = 0
    best_model = None
    best_features = []
    best_eq_info = None
    
    print("\nIncremental model building to reach target MSE:")
    print(f"Target MSE = {target_mse:.4f}")
    print("-" * 50)
    
    # Test with increasing numbers of top variables
    for n_vars in range(1, min(max_vars + 1, len(ranking) + 1)):
        # Select top n_vars from ranking
        selected_features = ranking[:n_vars]
        X_subset_train = X_train.iloc[:, selected_features]
        X_subset_test = X_test.iloc[:, selected_features]
        
        # Try different numbers of components
        for n_comp in range(1, min(n_vars + 1, 10)):
            # Scale data
            X_scaler = StandardScaler()
            X_train_scaled = X_scaler.fit_transform(X_subset_train)
            X_test_scaled = X_scaler.transform(X_subset_test)
            
            # Set up y-scaler
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
            
            # Create and fit PLS model
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_train_scaled, y_train_scaled)
            
            # Make predictions on test set
            y_pred_scaled = pls.predict(X_test_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
            
            # Calculate MSE and R²
            mse = np.mean((y_test - y_pred) ** 2)
            ss_total = np.sum((y_test - y_test.mean()) ** 2)
            ss_residual = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            
            print(f"Vars: {n_vars}, Components: {n_comp}, MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Check if this is better than our current best
            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_n_vars = n_vars
                best_components = n_comp
                best_model = pls
                best_features = selected_features
                
                # If we've hit our target MSE, we can stop
                if mse <= target_mse:
                    print(f"\nTarget MSE achieved with {n_vars} variables and {n_comp} components!")
                    print(f"MSE = {mse:.4f}, R² = {r2:.4f}")
                    
                    # Generate equation
                    variable_names = X.columns[selected_features].tolist()
                    eq_info = generate_empirical_equation(
                        X, y, selected_features, n_components=n_comp, variable_names=variable_names
                    )
                    best_eq_info = eq_info
                    return {
                        'n_variables': n_vars,
                        'variables': X.columns[selected_features].tolist(),
                        'n_components': n_comp,
                        'mse': mse,
                        'r2': r2,
                        'model': pls,
                        'features': selected_features,
                        'equation_info': eq_info
                    }
    
    # If we get here, we didn't achieve the target MSE
    print(f"\nCould not achieve target MSE of {target_mse:.4f} with {max_vars} variables.")
    print(f"Best model: {best_n_vars} variables, {best_components} components")
    print(f"MSE = {best_mse:.4f}, R² = {best_r2:.4f}")
    
    # Generate equation for best model
    variable_names = X.columns[best_features].tolist()
    eq_info = generate_empirical_equation(
        X, y, best_features, n_components=best_components, variable_names=variable_names
    )
    
    return {
        'n_variables': best_n_vars,
        'variables': X.columns[best_features].tolist(),
        'n_components': best_components,
        'mse': best_mse,
        'r2': best_r2,
        'model': best_model,
        'features': best_features,
        'equation_info': eq_info
    }

def validate_model_performance(X, y, model_result, n_splits=5):
    """
    Perform detailed validation of model performance using cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    model_result : dict
        Result from build_model_with_target_mse function
    n_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    Dictionary with validation results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    
    # Extract model details
    n_components = model_result['n_components']
    selected_features = model_result['features']
    X_subset = X.iloc[:, selected_features]
    
    # Set up cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store fold results
    fold_results = []
    predictions = []
    actuals = []
    
    # Perform cross-validation
    fold_idx = 1
    for train_idx, test_idx in kf.split(X_subset):
        # Split data
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        
        # Fit model
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_scaled = pls.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'mse': mse,
            'r2': r2
        })
        
        # Store predictions and actuals for later analysis
        for i in range(len(y_test)):
            predictions.append(y_pred[i])
            actuals.append(y_test.iloc[i])
        
        fold_idx += 1
    
    # Calculate aggregate statistics
    results_df = pd.DataFrame(fold_results)
    mean_mse = results_df['mse'].mean()
    std_mse = results_df['mse'].std()
    mean_r2 = results_df['r2'].mean()
    std_r2 = results_df['r2'].std()
    
    # Print summary
    print("\nCross-Validation Results:")
    print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    return {
        'fold_results': results_df,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'predictions': predictions,
        'actuals': actuals
    }

def plot_variable_contributions(model_result):
    """
    Plot the contribution of each variable to the prediction.
    
    Parameters:
    -----------
    model_result : dict
        Result from build_model_with_target_mse function
        
    Returns:
    --------
    Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get coefficient information
    eq_info = model_result['equation_info']
    raw_coefficients = eq_info['raw_coefficients']
    
    # Sort variables by absolute coefficient values
    sorted_vars = sorted(raw_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    var_names = [item[0] for item in sorted_vars]
    coef_values = [item[1] for item in sorted_vars]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(var_names, coef_values)
    
    # Color bars based on coefficient sign
    for i, bar in enumerate(bars):
        if coef_values[i] > 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    
    # Add intercept if available
    if 'equation_readable' in eq_info:
        intercept_str = eq_info['equation_readable'].split()[0]
        try:
            intercept = float(intercept_str)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.text(0.02, len(var_names) + 0.2, f"Intercept = {intercept:.4f}", fontsize=10)
        except ValueError:
            pass
    
    # Customize plot
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_ylabel('Variable', fontsize=12)
    ax.set_title('Variable Contributions to CH4 Prediction', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels to bars
    for i, v in enumerate(coef_values):
        if abs(v) >= 0.01:
            value_text = f"{v:.4f}"
        elif abs(v) >= 0.0001:
            value_text = f"{v:.6f}"
        else:
            value_text = f"{v:.8f}"
        
        if v > 0:
            ax.text(v + 0.001, i, value_text, va='center', fontsize=9)
        else:
            ax.text(v - 0.001, i, value_text, va='center', ha='right', fontsize=9)
    
    plt.tight_layout()
    return fig

def visualize_target_mse_model(X, y, optimal_model, output_dir):
    """
    Create a visualization for the target MSE empirical equation model.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    optimal_model : dict
        Result from build_model_with_target_mse function
    output_dir : str
        Directory to save the plot
    """
    # Extract model components
    equation_info = optimal_model['equation_info']
    pls = equation_info['pls_model']
    selected_features = equation_info['selected_features']
    X_scaler = equation_info['X_scaler']
    y_scaler = equation_info['y_scaler']
    
    # Convert y to numpy if it's a Series
    y_array = y.values if hasattr(y, 'values') else y
    
    # Make predictions
    X_selected = X.iloc[:, selected_features] if hasattr(X, 'iloc') else X[:, selected_features]
    X_scaled = X_scaler.transform(X_selected)
    y_pred_scaled = pls.predict(X_scaled)
    
    # Ensure prediction has correct shape for inverse transform (must be 2D)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Ensure y is flattened for plotting
    y_flat = y_array.flatten() if hasattr(y_array, 'flatten') else y_array
    
    # Calculate statistics
    mse = np.mean((y_flat - y_pred) ** 2)
    r2 = 1 - np.sum((y_flat - y_pred) ** 2) / np.sum((y_flat - np.mean(y_flat)) ** 2)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_flat, y_pred, alpha=0.7)
    
    # Add diagonal line (perfect prediction)
    min_val = min(np.min(y_flat), np.min(y_pred))
    max_val = max(np.max(y_flat), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Observed', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title(f'Target MSE Model: Observed vs Predicted (R² = {r2:.4f})', fontsize=14)
    
    # Add statistics to the plot
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, fontsize=11)
    plt.text(0.05, 0.90, f'MSE = {mse:.4f}', transform=plt.gca().transAxes, fontsize=11)
    plt.text(0.05, 0.85, f'Components = {equation_info["n_components"]}', transform=plt.gca().transAxes, fontsize=11)
    plt.text(0.05, 0.80, f'# Variables = {len(selected_features)}', transform=plt.gca().transAxes, fontsize=11)
    
    # Add the equation to the plot (truncate if too long)
    eq_text = equation_info['equation_readable']
    if len(eq_text) > 50:
        eq_text = eq_text[:47] + "..."
    plt.text(0.05, 0.75, f'Equation: {eq_text}', transform=plt.gca().transAxes, fontsize=9)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/target_mse_observed_vs_predicted.png', dpi=300)
    plt.close()
    
    # Create a residual plot (observed - predicted)
    residuals = y_flat - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Target MSE Model: Residual Plot', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the residual plot
    plt.savefig(f'{output_dir}/target_mse_residual_plot.png', dpi=300)
    plt.close()
    
    return plt.gcf()

def extract_equation_components(equation_str):
    """
    Extract all coefficients and variables from an equation string.
    
    Parameters:
    -----------
    equation_str : str
        Equation string in the format "a + b*X1 + c*X2 - d*X3 ..."
        
    Returns:
    --------
    dict: Dictionary with keys for 'intercept' and each variable
    """
    import re
    
    # Initialize the result dictionary
    coefficients = {}
    
    # Standardize the equation string for easier parsing
    # Replace minus with plus-minus for consistent splitting
    eq_normalized = equation_str.replace(' - ', ' + -')
    
    # Handle the special case where equation starts with a negative term
    if eq_normalized.startswith('-'):
        eq_normalized = '0 + ' + eq_normalized
    
    # Split by plus signs
    terms = eq_normalized.split(' + ')
    
    # Process each term
    for term in terms:
        term = term.strip()
        
        # Skip empty terms
        if not term:
            continue
            
        # Check if the term has a multiplication (*) which indicates a variable
        if '*' in term:
            # Split coefficient and variable
            parts = term.split('*', 1)  # Split only on the first '*'
            coef_str = parts[0].strip()
            var_name = parts[1].strip()
            
            # Convert coefficient to float
            try:
                coef_value = float(coef_str)
                coefficients[var_name] = coef_value
            except ValueError as e:
                print(f"Error converting coefficient '{coef_str}' to float: {e}")
        else:
            # This is likely the intercept
            try:
                # Convert to float and set as intercept
                intercept = float(term)
                coefficients['Intercept'] = intercept
            except ValueError as e:
                print(f"Error converting intercept '{term}' to float: {e}")
    
    return coefficients

def save_equation_coefficients(equation_info, filename):
    """
    Save the coefficients of an empirical equation to a CSV file.
    
    Parameters:
    -----------
    equation_info : dict
        Dictionary containing equation information from generate_empirical_equation
    filename : str
        Path to save the CSV file
    """
    import pandas as pd
    import os
    
    # Attempt to extract coefficients from raw_coefficients
    if 'raw_coefficients' in equation_info:
        coefficients = equation_info['raw_coefficients'].copy()
    else:
        coefficients = {}
    
    # Extract all coefficients including intercept from the equation string
    equation_str = equation_info['equation_readable']
    parsed_coeffs = extract_equation_components(equation_str)
    
    # Make sure we have the intercept
    if 'Intercept' in parsed_coeffs:
        intercept = parsed_coeffs['Intercept']
    else:
        print("Warning: Could not extract intercept from equation.")
        intercept = 0.0
    
    # Update coefficients with parsed values if needed
    for var, coef in parsed_coeffs.items():
        if var != 'Intercept':
            coefficients[var] = coef
    
    # Create a dataframe with coefficients (all variables first, then intercept)
    var_list = list(coefficients.keys())
    coef_list = list(coefficients.values())
    
    # Add intercept
    df = pd.DataFrame({
        'Variable': var_list + ['Intercept'],
        'Coefficient': coef_list + [intercept]
    })
    
    # Add equation metadata
    metadata = pd.DataFrame({
        'Metric': ['R-squared', 'MSE', 'Components', 'Variables', 'Equation'],
        'Value': [
            equation_info['r2'], 
            equation_info['mse'], 
            equation_info['n_components'],
            len(equation_info.get('selected_features', [])),
            equation_info['equation_readable']
        ]
    })
    
    # Save coefficients to CSV
    df.to_csv(filename, index=False)
    
    # Save metadata to a separate file
    metadata_filename = os.path.splitext(filename)[0] + '_metadata.csv'
    metadata.to_csv(metadata_filename, index=False)
    
    print(f"Saved coefficients to {filename}")
    print(f"Saved metadata to {metadata_filename}")
    
    return df
    
def export_to_excel(optimal_equation_info, target_mse_equation_info, filename):
    """
    Export both sets of equation coefficients to a single Excel file with multiple worksheets.
    Falls back to CSV files if Excel export fails.
    
    Parameters:
    -----------
    optimal_equation_info : dict
        Dictionary with optimal equation information
    target_mse_equation_info : dict
        Dictionary with target MSE equation information
    filename : str
        Path to save the Excel file
    """
    import pandas as pd
    import os
    
    try:
        # Try to use xlsxwriter first
        try:
            import xlsxwriter
            excel_engine = 'xlsxwriter'
        except ImportError:
            # If xlsxwriter is not available, try openpyxl
            try:
                import openpyxl
                excel_engine = 'openpyxl'
                print("xlsxwriter not found, using openpyxl instead.")
            except ImportError:
                # If neither is available, use default (could be 'xlwt' for xls files)
                excel_engine = None
                print("Neither xlsxwriter nor openpyxl found, using default engine.")
        
        # Extract coefficients for optimal equation
        optimal_coeffs = extract_equation_components(optimal_equation_info['equation_readable'])
        
        # Create dataframe for optimal equation coefficients
        df_optimal = pd.DataFrame({
            'Variable': list(optimal_coeffs.keys()),
            'Coefficient': list(optimal_coeffs.values())
        })
        
        # Move intercept to the bottom (for consistency)
        if 'Intercept' in df_optimal['Variable'].values:
            intercept_row = df_optimal[df_optimal['Variable'] == 'Intercept']
            df_optimal = pd.concat([
                df_optimal[df_optimal['Variable'] != 'Intercept'],
                intercept_row
            ]).reset_index(drop=True)
        
        # Extract coefficients for target MSE equation
        target_coeffs = extract_equation_components(target_mse_equation_info['equation_readable'])
        
        # Create dataframe for target MSE equation coefficients
        df_target = pd.DataFrame({
            'Variable': list(target_coeffs.keys()),
            'Coefficient': list(target_coeffs.values())
        })
        
        # Move intercept to the bottom (for consistency)
        if 'Intercept' in df_target['Variable'].values:
            intercept_row = df_target[df_target['Variable'] == 'Intercept']
            df_target = pd.concat([
                df_target[df_target['Variable'] != 'Intercept'],
                intercept_row
            ]).reset_index(drop=True)
        
        # Create metadata dataframe
        metadata = pd.DataFrame({
            'Metric': ['R-squared', 'MSE', 'Components', 'Variables'],
            'Optimal Equation': [
                optimal_equation_info['r2'],
                optimal_equation_info['mse'],
                optimal_equation_info['n_components'],
                len(optimal_equation_info.get('selected_features', []))
            ],
            'Target MSE Equation': [
                target_mse_equation_info['r2'],
                target_mse_equation_info['mse'],
                target_mse_equation_info['n_components'],
                len(target_mse_equation_info.get('selected_features', []))
            ]
        })
        
        # Create equations dataframe
        equations = pd.DataFrame({
            'Equation Type': ['Optimal Equation', 'Target MSE Equation'],
            'Equation': [
                optimal_equation_info['equation_readable'],
                target_mse_equation_info['equation_readable']
            ]
        })
        
        # Try to write to Excel
        try:
            with pd.ExcelWriter(filename, engine=excel_engine) as writer:
                df_optimal.to_excel(writer, sheet_name='Optimal Equation', index=False)
                df_target.to_excel(writer, sheet_name='Target MSE Equation', index=False)
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                equations.to_excel(writer, sheet_name='Full Equations', index=False)
                
                # Additional formatting with xlsxwriter
                if excel_engine == 'xlsxwriter':
                    # Get workbook and worksheet objects for formatting
                    workbook = writer.book
                    worksheet = writer.sheets['Full Equations']
                    
                    # Set the column width and add text wrapping for the equation column
                    worksheet.set_column('B:B', 80)
                    
                    # Add text wrapping format
                    wrap_format = workbook.add_format({'text_wrap': True})
                    worksheet.set_column('B:B', 80, wrap_format)
            
            print(f"Exported equation coefficients to Excel file: {filename}")
            return True
        
        except Exception as e:
            print(f"Failed to write Excel file: {e}")
            print("Falling back to CSV files...")
            
            # Fallback to CSV files
            base_filename = os.path.splitext(filename)[0]
            df_optimal.to_csv(f"{base_filename}_optimal_equation.csv", index=False)
            df_target.to_csv(f"{base_filename}_target_mse_equation.csv", index=False)
            metadata.to_csv(f"{base_filename}_metadata.csv", index=False)
            equations.to_csv(f"{base_filename}_full_equations.csv", index=False)
            
            print(f"Exported equation data to CSV files with base name: {base_filename}")
            return False
        
    except Exception as e:
        print(f"Error in export_to_excel: {e}")
        return False
    
def validate_against_original_results(results, original_r2=0.6377568):
    """Compare new results with original Unscrambler X results"""
    difference = abs(results['r2'] - original_r2)
    if difference < 0.05:
        print(f"✓ Results consistent with original (Δ = {difference:.4f})")
    else:
        print(f"⚠ Results differ from original (Δ = {difference:.4f})")

def generate_methodology_text():
    """Generate methodology text for paper"""
    methodology = """
    Partial Least Squares Regression (PLSR) analysis was performed using Python's 
    scikit-learn library. The analysis workflow included:
    
    1. Data standardization using StandardScaler
    2. Optimal component selection via 5-fold cross-validation
    3. Variable importance assessment using VIP scores
    4. Model validation through train-test splitting
    5. Correlation loadings visualization with Hotelling's T² ellipse
    
    The model's predictive performance was evaluated using R² and RMSE metrics.
    """
    return methodology

# Add at the top of main()
np.random.seed(42)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def model_diagnostics(X, y, equation_info):
    """Generate comprehensive model diagnostics for PLS model"""
    import scipy.stats as stats
    from scipy.stats import shapiro, jarque_bera
    
    # Extract model components
    pls = equation_info['pls_model']
    selected_features = equation_info['selected_features']
    X_scaler = equation_info['X_scaler']
    y_scaler = equation_info['y_scaler']
    
    # Get predictions and residuals
    X_selected = X.iloc[:, selected_features] if hasattr(X, 'iloc') else X[:, selected_features]
    X_scaled = X_scaler.transform(X_selected)
    y_pred_scaled = pls.predict(X_scaled)
    
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_array = y.values if hasattr(y, 'values') else y
    y_flat = y_array.flatten() if hasattr(y_array, 'flatten') else y_array
    
    residuals = y_flat - y_pred
    
    diagnostics = {}
    
    # 1. Durbin-Watson test for autocorrelation
    try:
        from statsmodels.stats.diagnostic import durbin_watson
        dw_stat = durbin_watson(residuals)
        diagnostics['durbin_watson'] = {
            'statistic': dw_stat,
            'interpretation': 'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'
        }
    except ImportError:
        # Simple autocorrelation if statsmodels not available
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        diagnostics['durbin_watson'] = {
            'autocorrelation': autocorr,
            'interpretation': 'No autocorrelation' if abs(autocorr) < 0.2 else 'Possible autocorrelation'
        }
    
    # 2. Normality test on residuals
    shapiro_stat, shapiro_p = shapiro(residuals)
    diagnostics['normality_test'] = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'is_normal': shapiro_p > 0.05,
        'interpretation': 'Residuals appear normal' if shapiro_p > 0.05 else 'Residuals may not be normal'
    }
    
    # 3. Homoscedasticity (constant variance) test
    # Breusch-Pagan test approximation
    squared_residuals = residuals ** 2
    correlation_with_fitted = np.corrcoef(y_pred, squared_residuals)[0, 1]
    diagnostics['homoscedasticity'] = {
        'correlation_fitted_squared_residuals': correlation_with_fitted,
        'is_homoscedastic': abs(correlation_with_fitted) < 0.3,
        'interpretation': 'Constant variance' if abs(correlation_with_fitted) < 0.3 else 'Non-constant variance detected'
    }
    
    # 4. Leverage points (high influence observations)
    # Calculate leverage using hat matrix diagonal
    try:
        hat_matrix = X_scaled @ np.linalg.pinv(X_scaled.T @ X_scaled) @ X_scaled.T
        leverage = np.diag(hat_matrix)
        high_leverage_threshold = 2 * X_scaled.shape[1] / X_scaled.shape[0]  # 2p/n rule
        high_leverage_points = np.where(leverage > high_leverage_threshold)[0]
        
        diagnostics['leverage_points'] = {
            'leverage_values': leverage,
            'threshold': high_leverage_threshold,
            'high_leverage_indices': high_leverage_points.tolist(),
            'n_high_leverage': len(high_leverage_points)
        }
    except Exception as e:
        diagnostics['leverage_points'] = {'error': f"Could not calculate leverage: {e}"}
    
    # 5. Outlier detection using standardized residuals
    std_residuals = residuals / np.std(residuals)
    outlier_threshold = 2.5  # |z| > 2.5
    outlier_indices = np.where(np.abs(std_residuals) > outlier_threshold)[0]
    
    diagnostics['outliers'] = {
        'standardized_residuals': std_residuals,
        'threshold': outlier_threshold,
        'outlier_indices': outlier_indices.tolist(),
        'n_outliers': len(outlier_indices),
        'outlier_values': y_flat[outlier_indices].tolist() if len(outlier_indices) > 0 else []
    }
    
    # 6. Overall model assessment
    diagnostics['model_summary'] = {
        'r_squared': equation_info['r2'],
        'mse': equation_info['mse'],
        'rmse': np.sqrt(equation_info['mse']),
        'n_components': equation_info['n_components'],
        'n_variables': len(selected_features),
        'n_observations': len(y_flat)
    }
    
    return diagnostics

def print_diagnostics_summary(diagnostics):
    """Print a summary of model diagnostics"""
    print("\n" + "="*50)
    print("MODEL DIAGNOSTICS SUMMARY")
    print("="*50)
    
    # Model performance
    summary = diagnostics['model_summary']
    print(f"Model Performance:")
    print(f"  R² = {summary['r_squared']:.4f}")
    print(f"  RMSE = {summary['rmse']:.4f}")
    print(f"  Components = {summary['n_components']}")
    print(f"  Variables = {summary['n_variables']}")
    
    # Diagnostics
    print(f"\nDiagnostic Tests:")
    print(f"  Autocorrelation: {diagnostics['durbin_watson']['interpretation']}")
    print(f"  Normality: {diagnostics['normality_test']['interpretation']}")
    print(f"  Homoscedasticity: {diagnostics['homoscedasticity']['interpretation']}")
    
    if 'leverage_points' in diagnostics and 'n_high_leverage' in diagnostics['leverage_points']:
        print(f"  High leverage points: {diagnostics['leverage_points']['n_high_leverage']}")
    
    print(f"  Outliers detected: {diagnostics['outliers']['n_outliers']}")
    
    print("="*50)

def get_rice_variable_mappings():
    """Map technical variable names to rice research terminology"""
    mappings = {
        'Δt·h': 'Flooding duration',
        'h·Ta': 'Water depth × Air temperature',
        'h·Tf': 'Water depth × Floodwater temperature', 
        'h·Ts': 'Water depth × Soil temperature',
        'Ta·Tf': 'Air × Floodwater temperature',
        'Ta·Ts': 'Air × Soil temperature',
        'SR·WS': 'Solar radiation × Wind speed',
        'WS': 'Wind speed',
        'SR': 'Solar radiation',
        'Ta': 'Air temperature',
        'Tf': 'Floodwater temperature',
        'Ts': 'Soil temperature',
        'h': 'Water depth',
        'Pa': 'Atmospheric pressure',
        'RH': 'Relative humidity'
    }
    return mappings

def translate_variable_names(equation_info):
    """Translate technical variable names to rice research terminology"""
    mappings = get_rice_variable_mappings()
    
    # Get the original equation
    original_equation = equation_info['equation_readable']
    translated_equation = original_equation
    
    # Replace variable names
    for tech_name, rice_name in mappings.items():
        # Use word boundaries to avoid partial replacements
        import re
        pattern = r'\b' + re.escape(tech_name) + r'\b'
        translated_equation = re.sub(pattern, rice_name, translated_equation)
    
    return {
        'original_equation': original_equation,
        'translated_equation': translated_equation,
        'variable_mappings': mappings
    }

def load_csv_with_encoding_detection(file_path):
    """
    Load CSV file with automatic encoding detection to handle various encodings.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    # List of common encodings to try
    encodings_to_try = [
        'utf-8',           # Default
        'windows-1252',    # Common Windows encoding
        'iso-8859-1',      # Latin-1
        'cp1252',          # Another name for windows-1252
        'utf-16',          # UTF-16
        'utf-8-sig'        # UTF-8 with BOM
    ]
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded with encoding: {encoding}")
            return data
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
            continue
        except Exception as e:
            print(f"Other error with encoding {encoding}: {e}")
            continue
    
    # If all encodings fail, try with error handling
    try:
        print("Trying with error='replace' to handle problematic characters...")
        data = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        print("Loaded with error replacement - some characters may be corrupted")
        return data
    except Exception as e:
        print(f"Final attempt failed: {e}")
        raise
        
def main():
    # Set output directory to the requested path
    output_dir = r"G:\2025\PLS2018UPLB\ANALYSIS\METADATA\NEWOUTS\COMPLETE_new"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Close any open figures to avoid warnings
    plt.close('all')
    
    # Load data - try to accommodate either local path or the original path
    data_path = "UYData_mod.csv"  # First try local path
    if not os.path.exists(data_path):
        data_path = r"G:\2025\PLS2018UPLB\ANALYSIS\METADATA\NEWDATA\UYData_mod.csv"  # Then try original path
    
    print(f"Loading data from: {data_path}")
    
    # Use the encoding detection function
    try:
        data = load_csv_with_encoding_detection(data_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Please check if the file exists and is not corrupted.")
        return None
    
    # Check the data shape
    print(f"Data shape: {data.shape}")
    print(f"All columns: {list(data.columns)}")
    
    # FIXED: Clean the data and properly select target column
    print("\nCleaning data and selecting target...")

    # First, remove string/object columns
    columns_to_remove = []
    for col in data.columns:
        if col.strip() == '' or data[col].dtype == 'object':  # Remove empty named or string columns
            columns_to_remove.append(col)
            print(f"Removing string/empty column: '{col}'")

    if columns_to_remove:
        data = data.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} string/empty columns")

    # Specifically target CH4 mg/m^2/h as requested
    target_column = "CH4 mg/m^2/h"

    if target_column not in data.columns:
        # Look for available CH4 columns
        ch4_columns = [col for col in data.columns if 'CH4' in str(col)]
        print(f"Target column '{target_column}' not found!")
        print(f"Available CH4 columns: {ch4_columns}")
        if ch4_columns:
            target_column = ch4_columns[0]  # Use the first available CH4 column
            print(f"Using '{target_column}' instead")
        else:
            print("No CH4 columns found! Stopping analysis.")
            return None

    # Extract target variable (CH4 mg/m^2/h)
    y = data[target_column].copy()
    print(f"Target variable: '{target_column}'")
    print(f"Target non-null values: {y.notna().sum()}/{len(y)}")

    # Extract predictor variables (remove both CH4 columns to avoid data leakage)
    ch4_columns = [col for col in data.columns if 'CH4' in str(col)]
    X = data.drop(columns=ch4_columns)
    print(f"Removed CH4 columns from predictors: {ch4_columns}")
    print(f"Predictor variables: {X.shape[1]} columns")

    # Remove rows where target is NaN
    valid_indices = y.notna()
    X_clean = X[valid_indices].copy()
    y_clean = y[valid_indices].copy()

    removed_rows = len(y) - len(y_clean)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing target values")

    # ROBUST: Check for completely empty columns first
    print(f"\nAnalyzing missing values in {X_clean.shape[1]} predictor columns...")
    completely_empty_cols = []
    partially_missing_cols = []

    for col in X_clean.columns:
        missing_count = X_clean[col].isna().sum()
        total_count = len(X_clean)
    
        if missing_count == total_count:  # Completely empty
            completely_empty_cols.append(col)
        elif missing_count > 0:  # Partially missing
            partially_missing_cols.append((col, missing_count, missing_count/total_count*100))

    # Remove completely empty columns
    if completely_empty_cols:
        print(f"Removing {len(completely_empty_cols)} completely empty columns:")
        for col in completely_empty_cols:
            print(f"  - {col}")
        X_clean = X_clean.drop(columns=completely_empty_cols)

    # Handle partially missing columns
    if partially_missing_cols:
        print(f"\nHandling {len(partially_missing_cols)} columns with partial missing data:")
        for col, count, pct in partially_missing_cols:
            print(f"  - {col}: {count} missing ({pct:.1f}%)")
        
            # Fill with median (now safe since column is not completely empty)
            median_val = X_clean[col].median()
            if pd.isna(median_val):  # Extra safety check
                print(f"    WARNING: Median is NaN for {col}, using 0 instead")
                X_clean[col] = X_clean[col].fillna(0)
            else:
                print(f"    Filled with median: {median_val:.4f}")
                X_clean[col] = X_clean[col].fillna(median_val)
    else:
        print("✓ No missing values in predictor columns")

    # Final validation
    x_nans = X_clean.isna().sum().sum()
    y_nans = y_clean.isna().sum()

    print(f"\nFinal validation:")
    print(f"  X missing values: {x_nans}")
    print(f"  y missing values: {y_nans}")

    if x_nans > 0 or y_nans > 0:
        print("ERROR: Still have NaN values after cleaning!")
    
        # Debug: Show which columns still have NaN
        if x_nans > 0:
            remaining_nan_cols = X_clean.columns[X_clean.isna().any()].tolist()
            print(f"Columns with remaining NaN: {remaining_nan_cols}")
            for col in remaining_nan_cols:
                nan_count = X_clean[col].isna().sum()
                print(f"  {col}: {nan_count} NaN values")
    
        return None

    print(f"\n✓ Data cleaning completed successfully")
    print(f"✓ Final dataset: X={X_clean.shape}, y={y_clean.shape}")
    print(f"✓ Target: {target_column}")
    print(f"✓ Sample range: {y_clean.min():.4f} to {y_clean.max():.4f}")

    # Update variables for the rest of the analysis
    X = X_clean
    y = y_clean
    
    print(f"\n" + "="*60)
    print("STARTING PLS ANALYSIS FOR RICE RIPENING")
    print("="*60)
    print(f"Target: {target_column}")
    print(f"Samples: {X.shape[0]}")
    print(f"Predictors: {X.shape[1]}")
    
    # Find optimal number of components
    print("\nFinding optimal number of components...")
    try:
        optimal_components, _ = find_optimal_components(X.values, y.values, max_components=min(10, X.shape[1]))
        print(f"Optimal number of components: {optimal_components}")
        plt.savefig(os.path.join(output_dir, 'optimal_components_selection.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error in component optimization: {e}")
        optimal_components = 3  # Fallback
        print(f"Using fallback: {optimal_components} components")
    
    # Create correlation loadings plot with optimal components
    print("Creating correlation loadings plots...")
    
    # PC1 vs PC2
    try:
        fig1, pls_model = create_correlation_loadings_plot(
            X, y.to_frame(target_column), 
            n_components=optimal_components,
            plot_components=(0, 1),
            title=f"PLS Correlation Loadings (PC1 vs PC2)\nRice Ripening - {target_column}"
        )
        fig1.savefig(os.path.join(output_dir, 'correlation_loadings_pc1_pc2.png'), dpi=300)
        plt.close()
        
        # PC1 vs PC3 if we have at least 3 components
        if optimal_components >= 3:
            fig2, _ = create_correlation_loadings_plot(
                X, y.to_frame(target_column),
                n_components=optimal_components,
                plot_components=(0, 2),
                title=f"PLS Correlation Loadings (PC1 vs PC3)\nRice Ripening - {target_column}"
            )
            fig2.savefig(os.path.join(output_dir, 'correlation_loadings_pc1_pc3.png'), dpi=300)
            plt.close()
        
        # PC2 vs PC3 if we have at least 3 components  
        if optimal_components >= 3:
            fig3, _ = create_correlation_loadings_plot(
                X, y.to_frame(target_column),
                n_components=optimal_components,
                plot_components=(1, 2),
                title=f"PLS Correlation Loadings (PC2 vs PC3)\nRice Ripening - {target_column}"
            )
            fig3.savefig(os.path.join(output_dir, 'correlation_loadings_pc2_pc3.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating correlation plots: {e}")
    
    # Analyze variable importance
    print("\nAnalyzing variable importance...")
    importance_analysis = analyze_variable_importance(X, y.values.reshape(-1, 1), n_components=optimal_components)
    
    # Add debugging
    print("Type of importance_analysis:", type(importance_analysis))
    if importance_analysis is None:
        print("WARNING: importance_analysis is None, creating default dictionary")
        # Create a default dictionary with VIP scores
        pls = PLSRegression(n_components=optimal_components)
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.values.reshape(-1, 1))
        pls.fit(X_scaled, y_scaled)
    
        vip = vip_scores(pls, X_scaled)
        importance_analysis = {
            'vip_scores': vip,
            'importance_ranking': np.argsort(vip)[::-1]
        }
    
    # Check importance
    if 'importance_ranking' not in importance_analysis:
        importance_analysis['importance_ranking'] = np.argsort(importance_analysis['vip_scores'])[::-1]
    
    # Print top 10 variables by importance
    print(f"\nTop 10 Variables by VIP Score for {target_column}:")
    for i, idx in enumerate(importance_analysis['importance_ranking'][:10]):
        print(f"{i+1}. {X.columns[idx]} - VIP Score: {importance_analysis['vip_scores'][idx]:.4f}")
    
    # Create a bar plot of variable importances (top 20)
    plt.figure(figsize=(12, 8))
    top_indices = importance_analysis['importance_ranking'][:20]
    plt.bar(X.columns[top_indices], importance_analysis['vip_scores'][top_indices])
    plt.title(f'Top 20 Variables by VIP Score\nRice Ripening - {target_column}')
    plt.xlabel('Variables')
    plt.ylabel('VIP Score')
    plt.xticks(rotation=90, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(output_dir, 'variable_importance_top20.png'), dpi=300)
    plt.close()
    
    # Select optimal variables
    print("\nSelecting optimal set of variables...")
    optimal_features, n_optimal, score_data = select_optimal_variables(
        X, y, max_features=20, importance_results=importance_analysis
    )
    plt.savefig(os.path.join(output_dir, 'optimal_feature_selection.png'), dpi=300)
    plt.close()
    
    print(f"\nOptimal number of variables: {n_optimal}")
    print("\nSelected variables:")
    for i, idx in enumerate(optimal_features):
        print(f"{i+1}. {X.columns[idx]} - VIP Score: {importance_analysis['vip_scores'][idx]:.4f}")
    
    # Generate empirical equation using optimal variables
    print("\nGenerating empirical equation...")
    equation_info = generate_empirical_equation(
        X, y, optimal_features, n_components=min(optimal_components, len(optimal_features)), variable_names=X.columns
    )
    
    # Print equation and performance metrics
    print(f"\nEmpirical Equation for {target_column} prediction:")
    print(equation_info['equation_readable'])
    print(f"\nR² = {equation_info['r2']:.4f}")
    print(f"MSE = {equation_info['mse']:.4f}")
    
    # Generate model diagnostics
    print("\nGenerating model diagnostics...")
    diagnostics = model_diagnostics(X, y, equation_info)
    print_diagnostics_summary(diagnostics)
    
    # Translate variable names for publication
    print("\nTranslating variable names...")
    translated = translate_variable_names(equation_info)
    print(f"\nOriginal equation: {translated['original_equation']}")
    print(f"Translated equation: {translated['translated_equation']}")

    # Save translated equation
    with open(os.path.join(output_dir, 'translated_equation.txt'), 'w', encoding='utf-8') as f:    
        f.write("RICE RESEARCH TERMINOLOGY EQUATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Technical: {translated['original_equation']}\n\n")
        f.write(f"Rice Research: {translated['translated_equation']}\n\n")
        f.write("Variable Mappings:\n")
        for tech, rice in translated['variable_mappings'].items():
            f.write(f"  {tech} = {rice}\n")
    
    # Plot observed vs predicted
    fig = plot_observed_vs_predicted(y, equation_info, X=X)
    fig.savefig(os.path.join(output_dir, 'observed_vs_predicted.png'), dpi=300)
    
    # Create a correlation heatmap of the selected variables with CH4
    selected_vars = X.columns[optimal_features].tolist()
    
    # Create correlation matrix
    corr_data = pd.concat([X[selected_vars], y], axis=1)
    correlation_matrix = corr_data.corr()
    
    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    
    # Add correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", 
                     ha='center', va='center', 
                     color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.title(f'Correlation Heatmap of Selected Variables with {target_column}')
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Save the equation and model details to a text file
    with open(os.path.join(output_dir, 'empirical_equation.txt'), 'w', encoding='utf-8') as f:
        f.write(f"PLS EMPIRICAL EQUATION FOR {target_column} PREDICTION\n")
        f.write("=" * 50 + "\n\n")
        f.write("EQUATION:\n")
        f.write(equation_info['equation_readable'] + "\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"R² = {equation_info['r2']:.6f}\n")
        f.write(f"MSE = {equation_info['mse']:.6f}\n")
        f.write(f"Number of PLS components = {equation_info['n_components']}\n")
        f.write(f"Number of variables used = {len(optimal_features)}\n\n")
        f.write("SELECTED VARIABLES:\n")
        for i, idx in enumerate(optimal_features):
            f.write(f"{i+1}. {X.columns[idx]} - VIP Score: {importance_analysis['vip_scores'][idx]:.6f}\n")
    
    # Save coefficients for the first equation
    save_equation_coefficients(equation_info, os.path.join(output_dir, 'optimal_equation_coefficients.csv'))
    
    # Compare models with different numbers of variables
    print("\nComparing models with different numbers of variables...")
    model_results = compare_variable_sets(
        X, y, importance_analysis, max_vars=20, n_components=None
    )

    # Plot results
    fig = plot_model_comparison(model_results)
    fig.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)

    # Generate text summary
    summary = generate_model_summary(model_results, X, importance_analysis)
    print(summary)

    # Save summary to file
    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Try to build a model with target MSE = 0.0090
    print("\nBuilding model with target MSE = 0.0090...")
    optimal_model = build_model_with_target_mse(
        X, y, importance_analysis, target_mse=0.0090, max_vars=20
    )
    
    # Validate model performance
    validation_results = validate_model_performance(X, y, optimal_model)

    # Plot variable contributions
    contrib_fig = plot_variable_contributions(optimal_model)
    contrib_fig.savefig(os.path.join(output_dir, 'variable_contributions.png'), dpi=300)

    # Print the optimized equation
    print(f"\nOptimized Empirical Equation for {target_column} prediction:")
    print(optimal_model['equation_info']['equation_readable'])
    print(f"\nR² = {optimal_model['r2']:.4f}")
    print(f"MSE = {optimal_model['mse']:.4f}")
    
    # Save coefficients for the target MSE equation
    save_equation_coefficients(optimal_model['equation_info'], 
                              os.path.join(output_dir, 'target_mse_equation_coefficients.csv'))
    
    # Export both sets of coefficients to a single Excel file
    export_to_excel(equation_info, optimal_model['equation_info'], 
                   os.path.join(output_dir, 'empirical_equations_coefficients.xlsx'))
    
    # Plot observed vs predicted for the target MSE model
    print("\nCreating visualization for target MSE model...")
    visualize_target_mse_model(X, y, optimal_model, output_dir)
    print(f"Visualization saved to {output_dir}/target_mse_observed_vs_predicted.png")
    
    print(f"\nAnalysis complete! All results saved to the '{output_dir}' directory.")
    print(f"Check '{os.path.join(output_dir, 'empirical_equation.txt')}' for the empirical equation details.")
    print(f"Equation coefficients saved to:")
    print(f"  - {os.path.join(output_dir, 'optimal_equation_coefficients.csv')}")
    print(f"  - {os.path.join(output_dir, 'target_mse_equation_coefficients.csv')}")
    print(f"  - {os.path.join(output_dir, 'empirical_equations_coefficients.xlsx')} (Excel file with both equations)")
    
    return equation_info

if __name__ == "__main__":
    main()