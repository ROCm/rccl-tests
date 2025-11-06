# Alternative Segmentation Algorithms for Performance Data

## Current Issues

The current heuristic-based segmentation algorithm:
- Can produce too many segments (up to 5 observed)
- Uses ad-hoc break detection based on local drops
- Lacks theoretical foundation
- May overfit to noise in transition regions

## Requirements

1. **Maximum 3 segments** - Interpretable and actionable
2. **Physical meaning** - Segments should correspond to actual performance regimes
3. **Robust** - Handle noise and outliers
4. **Automatic** - No manual tuning per benchmark

---

## Proposed Algorithm 1: K-Means Clustering on Performance Derivatives

### Concept

Cluster data points based on their **performance characteristics** rather than just time values. Use the derivative (rate of change) and absolute performance to identify regimes.

### Method

1. **Feature Engineering:**
   ```
   For each size point i:
     - y_i = TOP(size_i) = max(wall_time, kernel_mean + kernel_std)
     - slope_i = (y_{i+1} - y_{i-1}) / (size_{i+1} - size_{i-1})  [central difference]
     - normalized_time_i = y_i / max(y)
     - log_size_i = log2(size_i)
   
   Feature vector: [normalized_time_i, slope_i, log_size_i]
   ```

2. **K-Means Clustering:**
   ```
   - Apply K-Means with K=3 on feature vectors
   - Use standardized features (zero mean, unit variance)
   - Initialize with K-Means++ for stability
   ```

3. **Segment Ordering:**
   ```
   - Order clusters by median size
   - Assign consecutive size ranges to each cluster
   - Handle boundary smoothing if needed
   ```

4. **Fit Models:**
   ```
   For each segment:
     - Try linear: y = a + b·size_MB
     - Try log-linear: y = a + b·log2(size)
     - Choose model with better R²
   ```

### Advantages

- **Principled:** Well-established clustering method
- **Automatic:** No manual threshold tuning
- **Robust:** Handles noise through averaging in clusters
- **Interpretable:** Features have physical meaning (performance level, rate of change, scale)

### Disadvantages

- Requires tuning of feature weights
- May split natural segments if noise is high
- K-Means assumes spherical clusters (may not match data structure)

---

## Proposed Algorithm 2: Piecewise Linear Regression with Penalty

### Concept

Find the **optimal 2 breakpoints** that minimize fitting error while penalizing complexity. This is a constrained optimization problem.

### Method

1. **Exhaustive Search (for 2 breakpoints):**
   ```python
   best_score = -inf
   best_breaks = None
   
   # Try all possible pairs of breakpoints
   for i in range(2, n-4):  # First breakpoint
       for j in range(i+2, n-2):  # Second breakpoint
           # Fit 3 segments: [0:i], [i:j], [j:n]
           score = compute_score(data, [i, j])
           if score > best_score:
               best_score = score
               best_breaks = [i, j]
   ```

2. **Scoring Function:**
   ```
   score = -BIC = -k·log(n) + 2·log(likelihood)
   
   Where:
     - k = number of parameters (6 for 3 segments with linear fits)
     - n = number of data points
     - likelihood based on residual sum of squares
   
   BIC (Bayesian Information Criterion) penalizes complexity
   ```

3. **Alternative: AIC (Akaike Information Criterion):**
   ```
   AIC = 2k - 2·log(likelihood)
   
   Choose breakpoints that minimize AIC
   ```

4. **Fit Models:**
   ```
   For each of 3 segments:
     - Fit both linear and log-linear
     - Choose based on R² or cross-validation
   ```

### Advantages

- **Optimal:** Finds globally best 2 breakpoints for 3 segments
- **Principled:** Based on information theory (BIC/AIC)
- **Simple:** Easy to understand and implement
- **Guaranteed 3 segments:** By design

### Disadvantages

- O(n²) complexity for exhaustive search (acceptable for n≈30)
- Assumes piecewise structure exists
- May place breaks in noisy regions

---

## Proposed Algorithm 3: Change Point Detection (Bayesian)

### Concept

Use **Bayesian change point detection** to identify where the data-generating process changes. Limit to 2 change points (3 segments).

### Method

1. **Model:**
   ```
   Assume each segment has:
     - Different mean performance level
     - Different variance
     - Different trend (slope)
   
   Prior: Uniform over possible breakpoint locations
   ```

2. **Bayesian Inference:**
   ```python
   from ruptures import Pelt, Binseg
   
   # Use Binary Segmentation with 2 change points
   algo = Binseg(model="l2").fit(y)
   breakpoints = algo.predict(n_bkps=2)
   
   # Or use PELT (Pruned Exact Linear Time)
   algo = Pelt(model="rbf").fit(y)
   breakpoints = algo.predict(pen=penalty_value)
   ```

3. **Penalty Selection:**
   ```
   Use cross-validation or:
     penalty = log(n) * sigma²  [BIC-like]
   
   Where sigma² is estimated variance
   ```

4. **Segment Characterization:**
   ```
   For each segment:
     - Compute mean, variance, trend
     - Fit linear/log-linear model
     - Report confidence intervals
   ```

### Advantages

- **Statistically rigorous:** Based on Bayesian inference
- **Robust:** Handles outliers and noise well
- **Automatic:** Penalty can be set automatically
- **Fast:** PELT is O(n) with pruning

### Disadvantages

- Requires external library (ruptures)
- More complex to understand
- May need tuning of penalty parameter

---

## Proposed Algorithm 4: Elbow Method on Residual Variance

### Concept

Fit models with 1, 2, and 3 segments. Choose 3 segments if the **reduction in residual variance** justifies the complexity.

### Method

1. **Fit Multiple Models:**
   ```
   Model 1: Single linear fit across all data
     - RSS_1 = residual sum of squares
   
   Model 2: Two segments (1 breakpoint)
     - Find optimal breakpoint
     - RSS_2 = sum of RSS for both segments
   
   Model 3: Three segments (2 breakpoints)
     - Find optimal 2 breakpoints
     - RSS_3 = sum of RSS for all 3 segments
   ```

2. **Variance Reduction:**
   ```
   R²_1 = 1 - RSS_1/TSS
   R²_2 = 1 - RSS_2/TSS
   R²_3 = 1 - RSS_3/TSS
   
   Where TSS = total sum of squares
   ```

3. **Decision Rule:**
   ```
   If (R²_3 - R²_2) > threshold:
       Use 3 segments
   Else if (R²_2 - R²_1) > threshold:
       Use 2 segments
   Else:
       Use 1 segment
   
   Threshold = 0.1 (10% improvement required)
   ```

4. **Always Output 3 Segments:**
   ```
   If fewer segments justified:
     - Use 3 segments anyway (as required)
     - But mark some as "merged" or "weak boundary"
     - Report that fewer segments may be sufficient
   ```

### Advantages

- **Interpretable:** Based on variance explained
- **Flexible:** Can adapt if 3 segments aren't needed
- **Simple:** Easy to implement and understand
- **Visual:** Can plot elbow curve

### Disadvantages

- Threshold selection is somewhat arbitrary
- May force 3 segments when 2 would suffice
- Doesn't account for parameter count (unlike BIC/AIC)

---

## Recommended Approach

### Primary Recommendation: **Algorithm 2 (Piecewise Linear with BIC)**

**Rationale:**
1. **Optimal:** Finds globally best 2 breakpoints
2. **Principled:** BIC provides statistical justification
3. **Simple:** Easy to implement and explain
4. **Guaranteed 3 segments:** Meets requirement exactly
5. **No external dependencies:** Pure Python/NumPy/SciPy

**Implementation Priority:**
```
1st choice: Algorithm 2 (Piecewise Linear + BIC)
2nd choice: Algorithm 3 (Bayesian Change Point) - if willing to add dependency
3rd choice: Algorithm 1 (K-Means) - if feature engineering is acceptable
```

---

## Physical Interpretation of 3 Segments

For RCCL collective operations, 3 segments typically correspond to:

### Segment 1: **Latency-Dominated** (Small Messages)
- Size range: 8 bytes to ~1-10 KB
- Behavior: Flat or slowly increasing time
- Model: Often log-linear (y ≈ L₀ + c·log(size))
- Physics: Fixed overhead dominates (kernel launch, synchronization)

### Segment 2: **Transition** (Medium Messages)
- Size range: ~1 KB to ~1 MB
- Behavior: Increasing time, changing algorithm
- Model: Linear or log-linear
- Physics: Algorithm switching, cache effects, protocol changes

### Segment 3: **Bandwidth-Dominated** (Large Messages)
- Size range: ~1 MB to 1 GB
- Behavior: Linear scaling with size
- Model: Linear (y ≈ L₀ + size/B_eff)
- Physics: Network/memory bandwidth is bottleneck

This 3-segment model aligns with communication theory and hardware characteristics.

---

## Implementation Pseudocode (Algorithm 2)

```python
import numpy as np
from scipy.optimize import curve_fit

def segment_with_bic(sizes, times, n_segments=3):
    """
    Find optimal breakpoints for n_segments using BIC.
    Returns breakpoint indices and segment fits.
    """
    n = len(sizes)
    n_breaks = n_segments - 1
    
    # Exhaustive search for 2 breakpoints
    best_bic = np.inf
    best_breaks = None
    
    for i in range(2, n-4):
        for j in range(i+2, n-2):
            breaks = [i, j]
            bic = compute_bic(sizes, times, breaks)
            if bic < best_bic:
                best_bic = bic
                best_breaks = breaks
    
    # Fit models to each segment
    segments = []
    start = 0
    for end in best_breaks + [n]:
        seg_sizes = sizes[start:end]
        seg_times = times[start:end]
        
        # Try both linear and log-linear
        linear_fit = fit_linear(seg_sizes, seg_times)
        loglin_fit = fit_loglinear(seg_sizes, seg_times)
        
        # Choose better fit
        if linear_fit['r2'] > loglin_fit['r2']:
            segments.append(linear_fit)
        else:
            segments.append(loglin_fit)
        
        start = end
    
    return best_breaks, segments

def compute_bic(sizes, times, breaks):
    """Compute BIC for given breakpoints."""
    n = len(sizes)
    k = 2 * len(breaks) + 2  # 2 params per segment
    
    rss = 0
    start = 0
    for end in breaks + [n]:
        seg_times = times[start:end]
        seg_fit = fit_best(sizes[start:end], seg_times)
        rss += seg_fit['rss']
        start = end
    
    # BIC = n·log(RSS/n) + k·log(n)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic
```

---

## Testing Plan

1. **Implement Algorithm 2** as primary method
2. **Run on existing data** (12 benchmarks)
3. **Compare with current algorithm:**
   - Number of segments (should be exactly 3)
   - Breakpoint locations
   - R² values per segment
   - Visual inspection of fits
4. **Validate physically:**
   - Do segments align with expected regimes?
   - Are breakpoints at reasonable sizes?
   - Do formulas make sense?

---

## References

- Bayesian Information Criterion (BIC): Schwarz, 1978
- Change Point Detection: Killick et al., "Optimal Detection of Changepoints", 2012
- K-Means Clustering: MacQueen, 1967
- Piecewise Linear Regression: Muggeo, "Estimating regression models with unknown break-points", 2003

---

## Date
November 4, 2025

## Author
AI Assistant (Claude Sonnet 4.5)



