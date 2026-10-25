# Operations runbook

## Database failover

If the primary database is unavailable, promote the replica by running `pgctl promote` on the standby host, then update the connection string in the secrets store. The recovery time objective (RTO) for a database failover is 15 minutes.

The team reviewed the quarterly plan and agreed to revisit open questions in the next planning cycle. Facilities updated the visitor badge process and the shared calendar for meeting rooms. The archive of older documents is stored on the shared drive under the yearly folders.

## Log rotation

Logs are rotated daily with `logrotate -f` and kept for 14 days.

The team reviewed the quarterly plan and agreed to revisit open questions in the next planning cycle. The archive of older documents is stored on the shared drive under the yearly folders. The cafeteria menu rotates weekly and dietary notes are posted next to the counter.

## Certificate renewal

Certificates are renewed with `certbot renew --quiet` thirty days before they expire.

Please keep shared spaces tidy and report broken equipment to the facilities desk. The archive of older documents is stored on the shared drive under the yearly folders. Please keep shared spaces tidy and report broken equipment to the facilities desk.

## Cache flush

To clear the edge cache run `cdnctl purge --all`; expect a short latency increase afterwards.

The team reviewed the quarterly plan and agreed to revisit open questions in the next planning cycle. The team reviewed the quarterly plan and agreed to revisit open questions in the next planning cycle. Please keep shared spaces tidy and report broken equipment to the facilities desk.

## Queue drain

Before maintenance drain the job queue with `queuectl drain --timeout 300`.

Please keep shared spaces tidy and report broken equipment to the facilities desk. The team reviewed the quarterly plan and agreed to revisit open questions in the next planning cycle. Internal newsletters are published on the first Monday of each month.
