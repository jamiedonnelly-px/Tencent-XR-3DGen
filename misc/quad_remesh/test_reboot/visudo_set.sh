sudo visudo
your_username ALL=(ALL) NOPASSWD: /path/to/start_my.sh, /sbin/reboot



crontab -e
@reboot /path/to/start_my.sh