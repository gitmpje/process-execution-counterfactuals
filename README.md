# Process Execution Counterfactuals

## Development

`uv run pytest`

## Roadmap

### Functionality and performance

* Visualize counterfactual action in process execution graph.
* Apply ActionSet changes directly on HeteroData object
* PM4PY OCEL integration
  * Define process execution mask(s) on OCEL (DataFrames)
  * Construct HeteroData from OCEL (DataFrames)
  * Apply ActionSet changes on OCEL
* Multiprocessing setup
  * `To fix this issue, refer to the "Safe importing of main module"
section in <https://docs.python.org/3/library/multiprocessing.html>.`

### Code testing

* Improve test coverage
