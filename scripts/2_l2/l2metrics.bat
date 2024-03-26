@echo off

@REM set dest_name=%1
@REM set src_log=%1

set out_dir=%1
set src_dir=%1

set metrics=(performance, art_match, art_activation)

@REM set out_dir=work\results\l2metrics\l2metrics\%dest_name%
@REM set src_dir=work\results\l2metrics\logs\%src_log%

for %%m in %metrics% do (
    python -m l2metrics -p %%m -o %%m -O %out_dir%/%%m -l %src_dir%
)
