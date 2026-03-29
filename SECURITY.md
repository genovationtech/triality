# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Triality, please report it responsibly.

**Do not** open a public issue for security vulnerabilities.

**Contact:** connect@genovationsolutions.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Security Considerations

- Triality is a computational framework — it does not store user credentials or PII by default
- The FastAPI application should be deployed behind a reverse proxy (nginx, Caddy) in production
- Environment variables containing API tokens should never be committed to version control
- The `.env` file is excluded from git via `.gitignore`

---

*Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS*
*"We build systems that understand reality."*
