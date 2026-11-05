# NEWS

## 0.1.7

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.6 and made the package uninstallable on
  Julia 1.10/1.11.
- Allows NonconvexIpopt 0.6. Requires Julia 1.10+.
