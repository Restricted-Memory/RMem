# Evaluate RMem on VOT

## 1. Create VOT Workspace

```bash
mkdir vot_workspace
cd vot_workspace
vot initialize vots2025/votst
```

This command initializes the workspace and downloads the sequences.

Workspace structure after download:

```text
vot_workspace/
├── config.yaml
├── results/
├── sequences/
└── trackers.ini
```

## 2. Configure Tracker

Edit `vot_workspace/trackers.ini`:

```ini
[RMem]
label = RMem
protocol = traxpython
command = tools.eval_vot
paths = <your-path>/RMem/aot_plus/
env_PATH = <your-path>/RMem/aot_plus:${PATH}
env_RMEM_VOT_WORKSPACE = <your-path>/RMem/vot_workspace
```

`env_RMEM_VOT_WORKSPACE` is used for logs, debug masks, and annotations.

## 3. Test Tracker

Run the following commands inside `vot_workspace`.

```bash
vot test RMem
```

Expected output:

```text
Stopping tracker
@@TRAX:quit
Test concluded successfuly
```

## 4. Run Evaluation

```bash
vot evaluate RMem
```

## 5. Pack Results

```bash
vot pack RMem
```
