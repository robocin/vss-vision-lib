v4l2-ctl -d $1 -c exposure_auto_priority=0
v4l2-ctl -d $1 -c brightness=168
v4l2-ctl -d $1 -c contrast=128
v4l2-ctl -d $1 -c saturation=255
v4l2-ctl -d $1 -c white_balance_temperature_auto=1
v4l2-ctl -d $1 -c backlight_compensation=0
v4l2-ctl -d $1 -c exposure_auto=1
v4l2-ctl -d $1 -c exposure_absolute=172
v4l2-ctl -d $1 -c focus_auto=0
v4l2-ctl -d $1 -c focus_absolute=1