# LIBERO-Spatial local results — 2026. 06. 17. (수) 13:05:07 KST

## monolithic_act (`checkpoints/libero_spatial/monolithic_act/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
number of parameters: 85.95M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/monolithic_act/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 1/10 [pick up the black bowl between the plate and the r] : 90.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 75.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 5.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 90.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 40.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 40.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 25.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 75.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 70.0%

📊 LIBERO-Spatial [monolithic_act] avg success: 61.0%  | per-task: [90.0, 75.0, 5.0, 90.0, 40.0, 40.0, 100.0, 25.0, 75.0, 70.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_131851-2dl8opfd
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_monolithic_act
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/2dl8opfd
wandb: updating run metadata; uploading summary
wandb: uploading wandb-metadata.json; uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: uploading wandb-summary.json; uploading config.yaml
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 61
wandb:      libero_spatial/task_0 90
wandb:      libero_spatial/task_1 75
wandb:      libero_spatial/task_2 5
wandb:      libero_spatial/task_3 90
wandb:      libero_spatial/task_4 40
wandb:      libero_spatial/task_5 40
wandb:      libero_spatial/task_6 100
wandb:      libero_spatial/task_7 25
wandb:      libero_spatial/task_8 75
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_monolithic_act at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/2dl8opfd
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_131851-2dl8opfd/logs
```

## random_prune (`checkpoints/libero_spatial/random_prune/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
number of parameters: 85.95M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/random_prune/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 1/10 [pick up the black bowl between the plate and the r] : 70.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 45.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 0.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 80.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 10.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 80.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 0.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 60.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 65.0%

📊 LIBERO-Spatial [random_prune] avg success: 51.0%  | per-task: [70.0, 45.0, 0.0, 100.0, 80.0, 10.0, 80.0, 0.0, 60.0, 65.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: setting up run l0n8969f
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_133357-l0n8969f
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_random_prune
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/l0n8969f
wandb: updating run metadata; uploading summary
wandb: uploading summary
wandb: uploading wandb-metadata.json; uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 51
wandb:      libero_spatial/task_0 70
wandb:      libero_spatial/task_1 45
wandb:      libero_spatial/task_2 0
wandb:      libero_spatial/task_3 100
wandb:      libero_spatial/task_4 80
wandb:      libero_spatial/task_5 10
wandb:      libero_spatial/task_6 80
wandb:      libero_spatial/task_7 0
wandb:      libero_spatial/task_8 60
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_random_prune at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/l0n8969f
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_133357-l0n8969f/logs
```

## tome_clustering (`checkpoints/libero_spatial/tome_clustering/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
number of parameters: 85.95M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/tome_clustering/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 1/10 [pick up the black bowl between the plate and the r] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 45.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 70.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 80.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 80.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 15.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 60.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 80.0%

📊 LIBERO-Spatial [tome_clustering] avg success: 73.0%  | per-task: [100.0, 45.0, 70.0, 100.0, 80.0, 80.0, 100.0, 15.0, 60.0, 80.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: setting up run qqv3ccfq
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_134527-qqv3ccfq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_tome_clustering
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/qqv3ccfq
wandb: updating run metadata; uploading summary
wandb: uploading summary
wandb: uploading wandb-metadata.json; uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 73
wandb:      libero_spatial/task_0 100
wandb:      libero_spatial/task_1 45
wandb:      libero_spatial/task_2 70
wandb:      libero_spatial/task_3 100
wandb:      libero_spatial/task_4 80
wandb:      libero_spatial/task_5 80
wandb:      libero_spatial/task_6 100
wandb:      libero_spatial/task_7 15
wandb:      libero_spatial/task_8 60
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_tome_clustering at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/qqv3ccfq
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_134527-qqv3ccfq/logs
```

## promerge_only (`checkpoints/libero_spatial/promerge_only/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
number of parameters: 53.84M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/promerge_only/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 1/10 [pick up the black bowl between the plate and the r] : 45.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 40.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 85.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 95.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 20.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 15.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 10.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 85.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 45.0%

📊 LIBERO-Spatial [promerge_only] avg success: 54.0%  | per-task: [45.0, 40.0, 85.0, 95.0, 20.0, 15.0, 100.0, 10.0, 85.0, 45.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: setting up run bhgjqe7s
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_140046-bhgjqe7s
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_promerge_only
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/bhgjqe7s
wandb: updating run metadata; uploading summary
wandb: uploading wandb-metadata.json; uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: uploading history steps 0-0, summary
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 54
wandb:      libero_spatial/task_0 45
wandb:      libero_spatial/task_1 40
wandb:      libero_spatial/task_2 85
wandb:      libero_spatial/task_3 95
wandb:      libero_spatial/task_4 20
wandb:      libero_spatial/task_5 15
wandb:      libero_spatial/task_6 100
wandb:      libero_spatial/task_7 10
wandb:      libero_spatial/task_8 85
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_promerge_only at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/bhgjqe7s
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_140046-bhgjqe7s/logs
```

## thinkproprio (`checkpoints/libero_spatial/thinkproprio/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
number of parameters: 53.84M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/thinkproprio/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
Loading weights:   0%|          | 0/196 [00:00<?, ?it/s]Loading weights: 100%|██████████| 196/196 [00:00<00:00, 77467.36it/s]
[transformers] [1mCLIPTextModel LOAD REPORT[0m from: openai/clip-vit-base-patch32
Key                                                            | Status     |  | 
---------------------------------------------------------------+------------+--+-
vision_model.encoder.layers.{0...11}.self_attn.k_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.out_proj.bias   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm1.weight        | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.k_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc1.weight            | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.q_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm1.bias          | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.out_proj.weight | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.q_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm2.weight        | UNEXPECTED |  | 
text_projection.weight                                         | UNEXPECTED |  | 
logit_scale                                                    | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.v_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm2.bias          | UNEXPECTED |  | 
vision_model.post_layernorm.bias                               | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc2.weight            | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc2.bias              | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.v_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc1.bias              | UNEXPECTED |  | 
vision_model.embeddings.position_embedding.weight              | UNEXPECTED |  | 
vision_model.pre_layrnorm.weight                               | UNEXPECTED |  | 
vision_model.embeddings.class_embedding                        | UNEXPECTED |  | 
visual_projection.weight                                       | UNEXPECTED |  | 
vision_model.pre_layrnorm.bias                                 | UNEXPECTED |  | 
vision_model.post_layernorm.weight                             | UNEXPECTED |  | 
vision_model.embeddings.patch_embedding.weight                 | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Task 1/10 [pick up the black bowl between the plate and the r] : 75.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 70.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 90.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 70.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 0.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 30.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 45.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 10.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 55.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 50.0%

📊 LIBERO-Spatial [thinkproprio] avg success: 49.5%  | per-task: [75.0, 70.0, 90.0, 70.0, 0.0, 30.0, 45.0, 10.0, 55.0, 50.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: setting up run rmfk54uh
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_141624-rmfk54uh
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_thinkproprio
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/rmfk54uh
wandb: updating run metadata; uploading summary
wandb: uploading wandb-metadata.json; uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml
wandb: uploading history steps 0-0, summary
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 49.5
wandb:      libero_spatial/task_0 75
wandb:      libero_spatial/task_1 70
wandb:      libero_spatial/task_2 90
wandb:      libero_spatial/task_3 70
wandb:      libero_spatial/task_4 0
wandb:      libero_spatial/task_5 30
wandb:      libero_spatial/task_6 45
wandb:      libero_spatial/task_7 10
wandb:      libero_spatial/task_8 55
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_thinkproprio at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/rmfk54uh
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_141624-rmfk54uh/logs
```

## promerge_film (`checkpoints/libero_spatial/promerge_film/policy_best.ckpt`)
```
🔥 成功锁定 CUDA GPU 加速管线！
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
number of parameters: 53.84M
KL Weight 10
Successfully loaded checkpoint from checkpoints/libero_spatial/promerge_film/policy_best.ckpt (strict=False)
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
Loading weights:   0%|          | 0/196 [00:00<?, ?it/s]Loading weights: 100%|██████████| 196/196 [00:00<00:00, 78653.23it/s]
[transformers] [1mCLIPTextModel LOAD REPORT[0m from: openai/clip-vit-base-patch32
Key                                                            | Status     |  | 
---------------------------------------------------------------+------------+--+-
vision_model.encoder.layers.{0...11}.self_attn.q_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.k_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc2.weight            | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm1.bias          | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm2.weight        | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.v_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm2.bias          | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.v_proj.weight   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.out_proj.bias   | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc2.bias              | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.k_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.layer_norm1.weight        | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.q_proj.bias     | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc1.bias              | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.self_attn.out_proj.weight | UNEXPECTED |  | 
vision_model.encoder.layers.{0...11}.mlp.fc1.weight            | UNEXPECTED |  | 
vision_model.pre_layrnorm.weight                               | UNEXPECTED |  | 
logit_scale                                                    | UNEXPECTED |  | 
vision_model.embeddings.position_embedding.weight              | UNEXPECTED |  | 
text_projection.weight                                         | UNEXPECTED |  | 
visual_projection.weight                                       | UNEXPECTED |  | 
vision_model.pre_layrnorm.bias                                 | UNEXPECTED |  | 
vision_model.post_layernorm.weight                             | UNEXPECTED |  | 
vision_model.post_layernorm.bias                               | UNEXPECTED |  | 
vision_model.embeddings.class_embedding                        | UNEXPECTED |  | 
vision_model.embeddings.patch_embedding.weight                 | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Task 1/10 [pick up the black bowl between the plate and the r] : 55.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 2/10 [pick up the black bowl next to the ramekin and pla] : 40.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 3/10 [pick up the black bowl from table center and place] : 90.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 4/10 [pick up the black bowl on the cookie box and place] : 90.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 5/10 [pick up the black bowl in the top drawer of the wo] : 35.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 6/10 [pick up the black bowl on the ramekin and place it] : 10.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 7/10 [pick up the black bowl next to the cookie box and ] : 100.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 8/10 [pick up the black bowl on the stove and place it o] : 85.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 9/10 [pick up the black bowl next to the plate and place] : 85.0%
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
[Warning]: datasets path /home/imshen19/snap/ProMerge/external/LIBERO/libero/libero/../datasets does not exist!
  Task 10/10 [pick up the black bowl on the wooden cabinet and p] : 40.0%

📊 LIBERO-Spatial [promerge_film] avg success: 63.0%  | per-task: [55.0, 40.0, 90.0, 90.0, 35.0, 10.0, 100.0, 85.0, 85.0, 40.0]
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/imshen19/.netrc.
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
wandb: Currently logged in as: imshen19 (imshen19-yonsei-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: setting up run pmf55gaj
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:269: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/home/imshen19/snap/ProMerge/.venv/lib/python3.12/site-packages/wandb/analytics/sentry.py:280: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/imshen19/snap/ProMerge/wandb/run-20260617_143029-pmf55gaj
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run libero_eval_promerge_film
wandb: ⭐️ View project at https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: 🚀 View run at https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/pmf55gaj
wandb: updating run metadata; uploading summary
wandb: uploading requirements.txt; uploading wandb-summary.json; uploading config.yaml; uploading wandb-metadata.json
wandb: uploading history steps 0-0, summary
wandb: 
wandb: Run history:
wandb: libero_spatial/avg_success ▁
wandb:      libero_spatial/task_0 ▁
wandb:      libero_spatial/task_1 ▁
wandb:      libero_spatial/task_2 ▁
wandb:      libero_spatial/task_3 ▁
wandb:      libero_spatial/task_4 ▁
wandb:      libero_spatial/task_5 ▁
wandb:      libero_spatial/task_6 ▁
wandb:      libero_spatial/task_7 ▁
wandb:      libero_spatial/task_8 ▁
wandb:                         +1 ...
wandb: 
wandb: Run summary:
wandb: libero_spatial/avg_success 63
wandb:      libero_spatial/task_0 55
wandb:      libero_spatial/task_1 40
wandb:      libero_spatial/task_2 90
wandb:      libero_spatial/task_3 90
wandb:      libero_spatial/task_4 35
wandb:      libero_spatial/task_5 10
wandb:      libero_spatial/task_6 100
wandb:      libero_spatial/task_7 85
wandb:      libero_spatial/task_8 85
wandb:                         +1 ...
wandb: 
wandb: 🚀 View run libero_eval_promerge_film at: https://wandb.ai/imshen19-yonsei-university/ProMerge/runs/pmf55gaj
wandb: ⭐️ View project at: https://wandb.ai/imshen19-yonsei-university/ProMerge
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260617_143029-pmf55gaj/logs
```

