"""The Inngest client every function is registered against.

`is_production` is deliberately not passed. Left alone, the SDK reads
`INNGEST_DEV`: unset means Inngest Cloud, `INNGEST_DEV=1` means the local dev
server at 127.0.0.1:8288, and a URL points it wherever you say. So the same code
runs both ways and the switch is an environment variable rather than an edit.
"""

import inngest

client = inngest.Inngest(app_id="gh-issue")
