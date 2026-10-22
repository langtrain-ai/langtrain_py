"""
langtrain Python SDK — the training CLI is the npm package; this package ships
`langtrain-observe` (langtrain.observe.cli) for inspecting traces.

Authentication is handled by langtrain-sdk (npm):
  npx langtrain login        → browser OAuth → saves ~/.langtrain/credentials.json

Or set the environment variable directly:
  export LANGTRAIN_API_KEY=your_key

Python usage:
  from langtrain import FastLanguageModel, LangtrainClient, DatasetIntelligence
"""
