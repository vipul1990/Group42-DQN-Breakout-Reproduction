# Results and notebook assets

These files are derived from the executed HTML notebook supplied by Group 42.

- `training_episode_summary_from_notebook_output.csv` contains the 275 episode lines displayed in the notebook's training console output. It includes frame, episode, reward, and epsilon. It intentionally does **not** include a loss column because the console output does not print per-episode loss values.
- `evaluation_results.csv` records the five displayed evaluation rewards and the reported mean and standard deviation. All five evaluation episodes recorded a reward of zero. This means our reduced-budget experiment did not achieve effective performance as  the original paper's results during evaluation.
- `results_summary.csv` contains the final summary metrics shown by the notebook.
- `reward_curve.png` is the training plot embedded in the executed notebook.

The original Colab CSV and checkpoint files were not separately supplied as attached files here. The script in `src/breakout_dqn.py` can create fresh logs, a checkpoint, evaluation outputs, and a reward curve when run again; a fresh run may produce different outcomes.
