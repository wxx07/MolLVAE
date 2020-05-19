# Note

* `exp.old` setting:
  * KL loss weight linearly increases from 1e-4 (epoch 0) to 1e-3 (epoch 10).
  * Previous version of ladder sampling. Only for comparation.
* `exp` setting
  * Same KL weight annealer as `exp`
  * Updated ladder sampling.  
* `exp.e50-150` setting
  * Continue training from the last pt of `exp` up to 150 epoches.

