# Running Training in tmux

## Step 1: Install tmux (if not installed)
```bash
sudo apt-get update
sudo apt-get install -y tmux
```

## Step 2: Stop current training (if running)
```bash
pkill -f tf_implied_vol_small.py
```

## Step 3: Start training in tmux
```bash
cd /home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility
./start_training_tmux.sh
```

Or manually:
```bash
cd /home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility
tmux new-session -d -s implied_vol_training -c /home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility "source /home/tanveer/miniconda3/etc/profile.d/conda.sh && conda activate poorpeople-env-py310 && python tf_implied_vol_small.py"
```

## Step 4: Attach to tmux session
```bash
tmux attach-session -t implied_vol_training
```

## Step 5: Detach from tmux (keep training running)
Press: `Ctrl+B`, then `D`

## Useful tmux commands:
- List sessions: `tmux list-sessions`
- Attach to session: `tmux attach-session -t implied_vol_training`
- Kill session: `tmux kill-session -t implied_vol_training`
- View logs: `tail -f ImpliedVolatility/training_output.log`
