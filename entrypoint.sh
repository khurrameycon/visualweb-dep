#!/bin/bash
# Pre-create log files with correct ownership
mkdir -p /app/logs
if [ ! -f /app/logs/supervisord.log ]; then
	touch /app/logs/supervisord.log
fi
chown -R browseruse:browseruse /app/logs

# Start supervisord
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf

