# YNX - Anonymous P2P Chat

Send messages directly to anyone without servers, accounts, or surveillance.

## Install & Run

```bash
curl ynx.ch | sh
```

That's it. YNX sets up Tor, generates your identity, and starts the chat client automatically.

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

## Example Session

```
$ curl ynx.ch | sh
YNX Ready! Address: 2VmF8K9Xz4j3Nq7BpL5tDwR6eY1Mc8Hs

ynx> /chat 3KsR9Lm2Nx8Qw5Yr7Bt4Cv6Ez1Hj9Mp2
Starting chat with 3KsR9Lm2Nx8Qw5Yr7Bt4Cv6Ez1Hj9Mp2

you> hello
they> hey there
you> this is completely anonymous right?
they> yep, pure P2P over Tor
```

## Technical Details

- **Cryptography**: Ed25519 signatures for message authentication
- **Network**: Tor v3 hidden services for anonymity  
- **Dependencies**: Python 3, Tor, OpenSSL (auto-installed)
- **Platforms**: Linux, macOS

Ready to chat anonymously? Just run the one-liner.
