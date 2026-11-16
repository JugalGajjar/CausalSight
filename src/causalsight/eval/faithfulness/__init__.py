"""The counterfactual test suite, computed from saved per-item outputs of `cs-eval`.

report   blind gap, gold-evidence gap, object-track gap, cite gap and entity-track gap, each a paired
         difference in flip rates with a question-clustered bootstrap interval (`cs-faithfulness`)
splits   the diagnostics needed to read a gap: by gold answer, by citation hit, control overlap,
         video dependence, reverse flips, coverage, re-citation (`cs-faithfulness-splits`)
"""
