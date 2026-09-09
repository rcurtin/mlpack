## `DBSCAN`

The `DBSCAN` class implements DBSCAN ("Density Based Spatial Clustering of
Applications with Noise"), a clustering technique.  DBSCAN iteratively finds
localized high-density data regions, and groups connected high-density regions
into clusters.  Clusters produced by DBSCAN may have arbitrary shapes, and
points far away from any high-density region will be separately classified as
noise.

DBSCAN does not require the user to guess the number of clusters, and
does not make any assumptions on the shape of the data.

#### Simple usage example:

```c++
// Use DBSCAN to cluster random data and print the number of points that
// fall into each cluster.

// Create random dataset with two separated 10-dimensional Gaussians.
arma::mat dataset = arma::join_rows(
    arma::randn<arma::mat>(10, 1000) + 3.0,  // 1000 points from N(-3, 1).
    arma::randn<arma::mat>(10, 1000) - 3.0,  // 1000 points from N( 3, 1).
    arma::randn<arma::mat>(10, 1) + 20.0);   // One outlier "noise" point.

mlpack::DBSCAN dbscan(0.5, 10);                  // Step 1: create object.
arma::Row<size_t> assignments;
arma::mat centroids;
dbscan.Cluster(dataset, assignments, centroids); // Step 2: perform clustering.

// Print the number of clusters.
std::cout << "Found " << centroids.n_cols << " centroids." << std::endl;

// Print the number of points in each cluster.
for (size_t c = 0; c < centroids.n_cols; ++c)
{
  std::cout << " * Cluster " << c << " has " << arma::accu(assignments == c)
      << " points." << std::endl;
}

// Print the number of noise points.
std::cout << " * " << arma::accu(assignments == SIZE_MAX) << " points "
    << "classified as noise." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `DBSCAN` objects.
 * [`Cluster()`](#clustering): perform clustering.
 * [Other functionality](#other-functionality) for loading, saving, inspecting,
   and estimating the radius to use.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [mlpack clustering algorithms](../modeling.md#clustering)
 * [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 * [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)

### Constructors

 * `dbscan = DBSCAN(epsilon=0.5, minPoints=5, batchMode=true)`
   - Create a `DBSCAN` object with the specified parameters.
   - Clustering results are highly sensitive to the values of `epsilon` and
     `minPoints`; it is recommended to tune these parameters for your dataset!
     * See the notes below for more information on choosing these parameters.
   - mlpack's default [kd-tree](../core/trees/kdtree.md) dual-tree range search
     functionality is used for range search operations.

---

 * `dbscan = DBSCAN(epsilon, minPoints, batchMode, rangeSearch)`
 * `dbscan = DBSCAN(epsilon, minPoints, batchMode, rangeSearch, pointSelector)`
   - Create a `DBSCAN` object with the specified parameters, giving
     pre-instantiated `RangeSearch` and `OrderedPointSelection` objects.

---

 * `dbscan = DBSCAN<RangeSearchType>(epsilon=0.5, minPoints=5, batchMode=true)`
 * `dbscan = DBSCAN<RangeSearchType>(epsilon, minPoints, batchMode, rangeSearch)`
   - Create a `DBSCAN` object with the specified parameters, giving a
     pre-instantiated `RangeSearchType` object.
   - The `RangeSearchType` template parameter can be arbitrarily chosen and is
     described in the
     [advanced functionality section](#advanced-functionality-template-parameters).

---

 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(epsilon=0.5, minPoints=5, batchMode=true)`
 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(epsilon, minPoints, batchMode, rangeSearch)`
 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(epsilon, minPoints, batchMode, rangeSearch, pointSelector)`
   - Create a `DBSCAN` object with the specified parameters, giving
     pre-instantiated `RangeSearchType` and `PointSelectionPolicy` objects.
   - The `RangeSearchType` and `PointSelectionPolicy` template parameters can be
     arbitrarily chosen and is described in the
     [advanced functionality section](#advanced-functionality-template-parameters).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `epsilon` | `double` | Maximum distance between points that are a part of the same cluster. | `0.5` |
| `minPoints` | `size_t` | Minimum number of points within distance `epsilon`
for a point to be considered a 'core point'. | `5` |
| `batchMode` | `bool` | Whether to use batch-mode range search to find neighbors of points. | `true` |
| `rangeSearch` | [`RangeSearchType`](#advanced-functionality-template-parameters) |
| `pointSelector` | [`PointSelectionPolicy`](#advanced-functionality-template-parameters) |

***Notes:***

 - Clustering results are very sensitive to the settings of `epsilon` and
   `minPoints`!  The defaults for both of those are likely not correct for any
   dataset; *manual tuning and experimentation is generally necessary*.

 - If `epsilon` is too small, then no points will be considered a part of the
   same cluster and all points will be classified as noise.  If `epsilon` is too
   large, then all points will be classified as one cluster.

 - `minPoints` specifies the minimum number of neighboring points that a point
   must have to be the root of a cluster (e.g. a 'core point').  As this
   increases, the minimum number of points in a cluster also increases, but
   fewer points can be 'core points' that are the root of clusters.

 - Setting `batchMode` to `false` can keep memory usage lower, but at the
   potential cost of runtime slowdown.

### Clustering

 * `dbscan.Cluster(data, centroids)`
 * `dbscan.Cluster(data, assignments)`
 * `dbscan.Cluster(data, assignments, centroids)`
   - Cluster the given data, storing the resulting cluster centroids in
     `centroids`.
   - `centroids` will be set to size `data.n_rows` x `numClusters`, where
     `numClusters` is the number of clusters found by the mean shift algorithm.
   - The `i`th cluster centroid can be obtained with `clusters.col(i)`.

---

#### Clustering Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix holding the dataset to be clustered. | _(N/A)_ |
| `centroids` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix that centroids will be stored into. | _(N/A)_ |
| `assignments` | [`arma::Row<size_t>`](../matrices.md) | Vector to store cluster assignments for each point into. | _(N/A)_ |

***Notes***:

 * Different types can be used for `data` and `centroids` (e.g., `arma::fmat` or
   any dense matrix type implementing the Armadillo API).  The types of `data`
   and `centroids` must be the same.

### Other Functionality

 * A `DBSCAN` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

 * As an alternative to constructor parameters,
   - epsilon can be set with `dbscan.Epsilon(newEpsilon)`,
   - the minimum number of points for a core point can be set with
     `dbscan.MinPoints(newMinPoints)`, and
   - the batch mode setting can be set with `dbscan.BatchMode(newBatchMode)`.

 * `dbscan.RangeSearch()` returns a reference to the instantiated
    [`RangeSearchType`](#advanced-functionality-template-parameters) object used
    for range searching.

 * `dbscan.PointSelector()` returns a reference to the instantiated
    [`PointSelectionPolicy`](#advanced-functionality-template-parameters) object
    used to choose the first point of a new cluster.

### Simple Examples

Perform DBSCAN clustering on the satellite dataset and print the average
distance from each point to its assigned centroid, as well as the indices of any
noise point.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create DBSCAN object with default parameters and perform clustering.
mlpack::DBSCAN dbscan;
arma::mat centroids;
arma::Row<size_t> assignments;
ms.Cluster(dataset, assignments, centroids);

// Print the number of clusters.
std::cout << "MeanShift computed " << centroids.n_cols << " clusters."
    << std::endl;

// Compute the average distance from each point to its assigned centroid.
double sumDist = 0.0;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  sumDist += mlpack::EuclideanDistance::Evaluate(
      dataset.col(i), centroids.col(assignments[i]));
}
const double avgDist = sumDist / (double) dataset.n_cols;

std::cout << "Average distance from a point to its assigned centroid: "
    << avgDist << "." << std::endl;
```

---

Perform DBSCAN clustering on the wave energy farm dataset, setting `batchMode`
to `false` to save RAM usage during range searching.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat dataset;
mlpack::Load("wave_energy_farm_100.csv", dataset, mlpack::Fatal);

// Create DBSCAN object and set parameters.
mlpack::DBSCAN dbscan;

// Perform the clustering.
arma::mat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
std::cout << arma::accu(assignments == SIZE_MAX) << " points were classified "
    << "as noise." << std::endl;

// Save the centroids to disk.
mlpack::Save("wave_energy_centroids.csv", centroids);
```

---

Perform DBSCAN clustering on the cloud dataset, using 32-bit floating point
matrices to represent the data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::Load("cloud.csv", dataset, mlpack::Fatal);

// Create the MeanShift object using a TriangularKernel.
mlpack::DBSCAN dbscan();

// Perform clustering.
arma::fmat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
std::cout << " - " << arma::accu(assignments == SIZE_MAX) << " points were "
    << "classified as noise and not assigned to any cluster." << std::endl;
```

---

Perform DBSCAN clustering on the cloud dataset, using mlpack's `RangeSearch`
class with the [`CoverTree`](../core/trees/cover_tree.md_) for range search
operations.

```c++

```

### Advanced Functionality: Template Parameters

The `DBSCAN` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
MeanShift<RangeSearchType, PointSelectionPolicy>
```

 * `RangeSearchType` (default ) ...

 * `PointSelectionPolicy` (default ) ...

---
