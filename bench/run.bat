@echo off

setlocal
setlocal enabledelayedexpansion

::# Put your dll in current directory 
::# or append its path to following variable use ';' as delimeter
set PATH=%PATH%;C:\yourBinPath

::# # # #  replace your load config file and 3D model HERE  # # # # # 
set modelfile=E:\projectBatch\robtop\windows\bench\Kitten.obj
set configfile=E:\projectBatch\robtop\windows\bench\config31.json
::# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

::# # # # Parameter setting below # # # # #  
::> Workmode (nscf/wscf/nsff/wsff)
set mode=wsff
::> filter radius
set sens_radius=3
::> grid resolution along longest axis
set reso=511
::> damp ratio mu
set damp_ratio=0.5
::> density change limit 
set design_step=0.06
::> volume reduction (not using)
set vol_red=0.05
::> whether to output density field in each iteration('[log/nolog]density')
set logrho=logdensity
::> whether to output compliance field in each iteration 
set logc=nologcompliance
::> power penalty coefficient
set power=3
::> volume threshold
set vol_thres=0.35
::> set shell width 
set shell_width=2
::> set minimal density
set min_rho=1e-3
::> testname (None for no testing)
set testname=None

start /b /wait .\robtop      ^
-jsonfile=!configfile!       ^
-meshfile=!modelfile!        ^
-outdir=.\result\            ^
-power_penalty=!power!       ^
-volume_ratio=!vol_thres!    ^
-filter_radius=!sens_radius! ^
-gridreso=!reso!             ^
-damp_ratio=!damp_ratio!     ^
-shell_width=!shell_width!   ^
-workmode=!mode!             ^
-design_step=!design_step!   ^
-vol_reduction=!vol_red!     ^
-poisson_ratio=0.4           ^
-min_density=!min_rho!       ^
-!logrho!                    ^
-!logc!                      ^
-testname=!testname!              


