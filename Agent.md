# Project Instructions

This project is built around a config-driven pipeline in `classic_methods/`. Contributors can add new preprocessing, embedding, clustering, similarity, or model stages, but every new stage must be made compatible with the pipeline contract and config tree.

## Non-negotiable rules

1. For every function, add a short comment immediately after its definition describing inputs and outputs.
2. For every implementation, add or update a test script, run it, and iterate until it passes.
3. When adding a new stage, do not only add the Python class. Also update the contract, config, and tests so it can actually run inside the pipeline.

## Current config structure

Only these top-level config entries should exist:

- `classic_methods/config/base.yaml`
- `classic_methods/config/paths.yaml`
- `classic_methods/config/recommendation/`
- `classic_methods/config/validation/`

Key files:

- `classic_methods/config/base.yaml`
  This is the orchestration entrypoint.
- `classic_methods/config/recommendation/pipeline.yaml`
  Main recommendation pipeline.
- `classic_methods/config/recommendation/pipeline_validation.yaml`
  Recommendation pipeline used inside validation runs.
- `classic_methods/config/validation/validation.yaml`
  Validation loop settings.

## Pipeline contract overview

The pipeline implementation is in:

- `classic_methods/src/pipeline.py`
- `classic_methods/src/pipeline_contracts.py`

Important artifact kinds already used by the project:

- `dataframe`
- `matrix`
- `similarity_matrix`
- `cluster_labels`
- `recommendation_scores`
- `index_array`
- `mapping`
- `model`

Every stage should expose compatibility information through a `StageContract` or equivalent class attributes such as:

- `input_type`
- `output_type`
- `input_artifacts`
- `output_artifacts`
- `contract`

If a stage consumes or produces multiple named artifacts, use `input_artifacts` and `output_artifacts`.

If a stage exports results through `export_artifacts()`, those artifact names must match the names used by downstream configs.

## What a contributor must do when adding a new function or class that will be used as a stage

the agent should handle all of the following:

1. Put the implementation in the correct module tree.
   Examples:
   - embeddings go under `classic_methods/src/embeddings/`
   - downstream recommendation or similarity models go under `classic_methods/src/models/`

2. Declare the pipeline input/output contract.
   The new class must clearly state:
   - what artifact names it expects
   - what artifact kinds those inputs are
   - what artifacts it returns
   - what artifact kinds those outputs are

3. Make exported artifact names match pipeline config names.
   Example:
   - if downstream config expects `embedding`, the stage must export `embedding`
   - if downstream config expects `model`, the stage must export `model`

4. Add or update package exports if needed.
   Typical places:
   - `classic_methods/src/embeddings/__init__.py`
   - `classic_methods/src/models/__init__.py`
   - `classic_methods/src/models/similarity_matrix/__init__.py`

5. Add a config file in the correct config subtree.
   Examples:
   - `classic_methods/config/recommendation/embeddings/<name>.yaml`
   - `classic_methods/config/recommendation/models/similarity_matrix/<name>.yaml`

6. Wire the config into the appropriate pipeline config when requested.
   Typical files:
   - `classic_methods/config/recommendation/pipeline.yaml`
   - `classic_methods/config/recommendation/pipeline_validation.yaml`

7. Add a regression test that proves:
   - the stage can be instantiated from config
   - the pipeline accepts its declared input/output contract
   - the produced artifacts have the expected names and shapes/types

## How to think about compatibility

Before changing configs, the agent must verify all of these:

1. The new stage’s input artifact names match the previous stage’s output artifact names.
2. The new stage’s input artifact kinds match the previous stage’s output artifact kinds.
3. The next stage’s expected inputs match the new stage’s outputs.
4. The selected pipeline method name exists and is callable.
5. If the stage is used in validation mode, it remains compatible with masked-data runs.

The agent should inspect:

- `classic_methods/src/pipeline_contracts.py`
- the previous stage class
- the next stage class
- the target YAML stage config
- the pipeline YAML that references it

The agent should not guess artifact names. It should align them exactly.

## Validation-specific note

Validation is orchestrated separately from the main recommendation pipeline.

- `Pipeline`
  Runs the configured recommendation stages and can return artifacts and ranked recommendations.
- `ValidationPipeline`
  Samples users, masks rows, calls `Pipeline.run(...)`, computes metrics, loops over repetitions, and writes one JSON summary.

If a contributor changes embeddings or models, the agent must ensure they still work in:

- `classic_methods/config/recommendation/pipeline.yaml`
- `classic_methods/config/recommendation/pipeline_validation.yaml`

Validation uses masked input data and expects the recommendation pipeline to run with:

- `save_artifacts=False`
- `return_recommendations=True`

## What to tell the agent

When a contributor adds a new stage, they should be able to say something like:

`I added a new embedding/model class in <path>. Make it pipeline-compatible: declare the correct input/output artifacts, add the config under config/recommendation, wire it into the right pipeline, and update tests so the pipeline validates and runs.`

If they want to replace an existing stage, they can say:

`Replace stage<N> with my new <embedding/model> and adjust contracts, exported artifact names, downstream configs, and tests so both the recommendation pipeline and validation pipeline remain compatible.`

## Minimum acceptance checklist for agent work

Before finishing, the agent should verify:

1. The new class is importable from its config `_target_`.
2. `Pipeline(...)` validates successfully with the changed config.
3. The stage exports the artifact names expected by downstream stages.
4. The relevant regression scripts pass.
5. If the stage participates in recommendation, `Pipeline.run(..., return_recommendations=True)` still works.
6. If the stage participates in validation, `ValidationPipeline` still works with the updated recommendation pipeline.
