# sortr

## Install
```bash
pip install -e .
```

## List registered tasks
```bash
python -c "import mjlab.tasks; print('\n'.join(mjlab.tasks.list_tasks()))"
```

## Play / Train
```bash
play  <task-id> --agent zero --viewer viser
train <task-id> --num_envs 4096
```
