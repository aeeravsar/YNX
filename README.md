# YNX - Anonymous P2P Chat

Send messages directly to anyone without servers, accounts, or surveillance.

## Install & Run

```bash
curl ynx.ch | sh
```

That's it. YNX sets up Tor, generates your identity, and starts the chat client automatically.

## Example Session

[![asciicast](https://asciinema.org/a/eWb5feGJN7U8RzGnlSVbEzzU8.svg)](https://asciinema.org/a/eWb5feGJN7U8RzGnlSVbEzzU8)

## How to Chat

1. Share your YNX address with someone
2. Use `/chat <their_address>` to start messaging
3. Messages are cryptographically signed and routed through Tor

## Commands

- `/chat <address>` - Start chatting
- `/add <address> <name>` - Save a contact
- `/contacts` - List contacts
- `/whoami` - Show your address
- `/exit` - Quit

## What Makes It Different

**No servers** - Messages go directly between users via Tor hidden services  
**No registration** - Your identity is a cryptographic keypair, nothing more  
**No metadata** - Tor routing hides who talks to whom  
**No permanence** - Use temp profiles that vanish when you quit

## Technical Details

- **Cryptography**: Ed25519 signatures for message authentication
- **Network**: Tor v3 hidden services for anonymity  
- **Dependencies**: Python 3, Tor, OpenSSL (auto-installed)
- **Platforms**: Linux, macOS

Ready to chat anonymously? Just run the one-liner.
