#!/usr/bin/env bash
# hpc/watch_and_submit_ablation.sh
# ======================================================================
# Polls until DDPM job finishes, then auto-submits ablation_steps.sh.
# Run this in a screen/tmux session on Explorer AFTER submitting DDPM.
#
# Usage (on Explorer, inside screen):
#   screen -S ablation_watcher
#   bash hpc/watch_and_submit_ablation.sh 5844256
#   Ctrl+A, D  (detach)

set -euo pipefail

DDPM_JOB_ID="${1:-}"
if [ -z "$DDPM_JOB_ID" ]; then
  echo "Usage: $0 <ddpm_job_id>"
  echo "  e.g. $0 5844256"
  exit 1
fi

PROJECT=/scratch/gupta.yashv/diffusion_policy
LOG="$PROJECT/logs/slurm/ablation_watcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Watching DDPM job $DDPM_JOB_ID ..."
log "Will submit ablation_steps.sh when job completes."

while true; do
  STATE=$(squeue -j "$DDPM_JOB_ID" --noheader --format='%T' 2>/dev/null || echo "DONE")

  if [[ "$STATE" == "DONE" || -z "$STATE" ]]; then
    log "DDPM job $DDPM_JOB_ID finished."

    CKPT="$PROJECT/checkpoints/ddpm_300ep/best.pt"
    if [ ! -f "$CKPT" ]; then
      log "WARNING: best.pt not found at $CKPT — checking epoch checkpoint ..."
      CKPT=$(ls "$PROJECT/checkpoints/ddpm_300ep/"epoch_*.pt 2>/dev/null | sort | tail -1)
      if [ -z "$CKPT" ]; then
        log "ERROR: No DDPM checkpoint found. Did training succeed? Check logs/slurm/ddpm_${DDPM_JOB_ID}.out"
        exit 1
      fi
      log "Using checkpoint: $CKPT"
    fi

    log "Submitting ablation_steps.sh ..."
    cd "$PROJECT"
    ABL_JOB=$(CKPT="$CKPT" sbatch --parsable hpc/ablation_steps.sh)
    log "Ablation job submitted: $ABL_JOB"

    # Update job_ids.txt
    echo "ABLATION_JOB=$ABL_JOB" >> "$PROJECT/logs/job_ids.txt"

    # Commit to git
    git -C "$PROJECT" add logs/job_ids.txt
    git -C "$PROJECT" commit -m "Auto-submit ablation job $ABL_JOB after DDPM $DDPM_JOB_ID completed"

    log "Done. Monitor ablation: squeue -j $ABL_JOB"
    log "  Or: tail -f $PROJECT/logs/slurm/ablation_${ABL_JOB}.out"
    break
  fi

  log "Job $DDPM_JOB_ID is $STATE — checking again in 10 min ..."
  sleep 600
done
