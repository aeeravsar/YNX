#!/bin/bash

# YNX - One-Liner P2P Messaging
# Usage: curl ynx.ch | sh

set -e


echo "YNX - One-Liner P2P Messaging"
echo "=============================================="

# Detect system
detect_system() {
    case "$OSTYPE" in
        linux-gnu*)
            if command -v apt-get >/dev/null 2>&1; then
                DISTRO="debian"
            elif command -v yum >/dev/null 2>&1; then
                DISTRO="redhat"
            elif command -v pacman >/dev/null 2>&1; then
                DISTRO="arch"
            elif command -v apk >/dev/null 2>&1; then
                DISTRO="alpine"
            else
                DISTRO="unknown"
            fi
            OS="linux"
            ;;
        darwin*)
            OS="macos"
            DISTRO="macos"
            ;;
        *)
            OS="unknown"
            DISTRO="unknown"
            ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install packages based on distro
install_packages() {
    packages="$*"
    
    case "$DISTRO" in
        "debian")
            echo "Installing packages: $packages"
            sudo apt-get update >/dev/null 2>&1
            sudo apt-get install -y $packages
            ;;
        "redhat")
            echo "Installing packages: $packages"
            sudo yum install -y $packages
            ;;
        "arch")
            echo "Installing packages: $packages"
            sudo pacman -S --noconfirm $packages
            ;;
        "alpine")
            echo "Installing packages: $packages"
            sudo apk add $packages
            ;;
        "macos")
            if command_exists brew; then
                echo "Installing packages via Homebrew: $packages"
                brew install $packages
            else
                echo "Error: Homebrew not found. Please install Homebrew first:"
                echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            ;;
        *)
            echo "Error: Unsupported distribution. Please install manually: $packages"
            exit 1
            ;;
    esac
}

# Check and install dependencies
check_dependencies() {
    echo "Checking system dependencies..."
    
    detect_system
    echo "Detected system: $OS ($DISTRO)"
    
    missing_packages=""
    
    # Check Python 3
    if ! command_exists python3; then
        case "$DISTRO" in
            "debian") missing_packages="$missing_packages python3" ;;
            "redhat") missing_packages="$missing_packages python3" ;;
            "arch") missing_packages="$missing_packages python" ;;
            "alpine") missing_packages="$missing_packages python3" ;;
            "macos") missing_packages="$missing_packages python3" ;;
        esac
    fi
    
    # Check Tor
    if ! command_exists tor; then
        missing_packages="$missing_packages tor"
    fi
    
    
    # Check openssl
    if ! command_exists openssl; then
        case "$DISTRO" in
            "debian") missing_packages="$missing_packages openssl" ;;
            "redhat") missing_packages="$missing_packages openssl" ;;
            "arch") missing_packages="$missing_packages openssl" ;;
            "alpine") missing_packages="$missing_packages openssl" ;;
            "macos") missing_packages="$missing_packages openssl" ;;
        esac
    fi
    
    
    
    # Install missing packages
    if [ "$missing_packages" != "" ]; then
        echo "Missing packages detected: $missing_packages"
        
        if [ "$OS" = "linux" ]; then
            if [ "$EUID" -eq 0 ]; then
                install_packages $missing_packages
            else
                echo "Sudo access required to install packages."
                echo "Please run: sudo $0"
                exit 1
            fi
        else
            install_packages $missing_packages
        fi
        
        echo "Dependencies installed successfully!"
    else
        echo "All dependencies satisfied!"
    fi
}

# Main execution
main() {
    check_dependencies
    
    echo "Starting YNX client..."
    echo ""
    
    # Create temporary Python script with unique name
    RANDOM_SUFFIX=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
    TEMP_SCRIPT="/tmp/ynx_${RANDOM_SUFFIX}.py"
    
    cat > "$TEMP_SCRIPT" << 'PYTHON_SCRIPT_END'
#!/usr/bin/env python3

import asyncio
import curses
import os
import sys
import signal
import subprocess
import json
import base64
import hashlib
import hmac
import binascii
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime

def socks5_send_receive(target_onion: str, message: str, socks_port: int, timeout: float = 10.0) -> tuple:
    """Send message and receive response via SOCKS5"""
    import socket
    import struct
    import os
    
    sock = None
    debug_messages = []
    
    try:
        debug_messages.append(f"Connecting to {target_onion} via proxy port {socks_port}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Connect to SOCKS proxy
        sock.connect(('127.0.0.1', socks_port))
        debug_messages.append("Connected to proxy")
        
        # SOCKS5 handshake
        sock.send(b'\x05\x01\x00')
        response = sock.recv(2)
        if len(response) != 2 or response[0] != 5 or response[1] != 0:
            debug_messages.append(f"Handshake failed: {response.hex() if response else 'no response'}")
            return "", debug_messages
        
        debug_messages.append("Handshake successful")
        
        # Send connection request
        domain_data = target_onion.encode()
        addr_data = struct.pack('!BBB', 5, 1, 0)
        addr_data += struct.pack('!B', 3)
        addr_data += struct.pack('!B', len(domain_data)) + domain_data
        addr_data += struct.pack('!H', 2323)
        sock.send(addr_data)
        
        debug_messages.append("Sent connection request")
        
        # Read response
        response = sock.recv(10)
        if len(response) < 4 or response[0] != 5 or response[1] != 0:
            debug_messages.append(f"Connection failed: {response.hex() if response else 'no response'}")
            return "", debug_messages
        
        debug_messages.append("Connected to target")
        
        # Send message
        sock.send(message.encode() + b'\n')
        debug_messages.append(f"Sent message: {message}")
        
        # Receive response
        sock.settimeout(5.0)
        response = b""
        while True:
            try:
                data = sock.recv(1024)
                if not data:
                    break
                response += data
                debug_messages.append(f"Received chunk: {data}")
                if b'\n' in response:
                    break
            except socket.timeout:
                debug_messages.append("Timeout waiting for response")
                break
        
        result = response.decode().strip()
        debug_messages.append(f"Final response: '{result}'")
        return result, debug_messages
        
    except Exception as e:
        debug_messages.append(f"Exception: {e}")
        return "", debug_messages
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass

# YNX custom alphabet for base58 encoding
YNX_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

@dataclass
class YNXProfile:
    name: str
    private_key: str
    public_key: str
    onion_address: str
    ynx_address: str

@dataclass
class VerifiedKey:
    public_key: str
    onion_address: str
    verified_at: float
    last_seen: float
    verification_status: str  # 'verified', 'pending', 'failed'

class YNXKeyCache:
    """Manages ephemeral cache of verified public keys"""
    
    def __init__(self):
        self.verified_keys: Dict[str, VerifiedKey] = {}
        self.pending_verifications: Dict[str, float] = {}  # Track pending key requests
    
    def add_verified_key(self, onion_address: str, public_key: str) -> None:
        """Add a verified public key to the cache"""
        self.verified_keys[onion_address] = VerifiedKey(
            public_key=public_key,
            onion_address=onion_address,
            verified_at=time.time(),
            last_seen=time.time(),
            verification_status='verified'
        )
    
    def get_public_key(self, onion_address: str) -> Optional[str]:
        """Get cached public key for an onion address"""
        if onion_address in self.verified_keys:
            key_info = self.verified_keys[onion_address]
            key_info.last_seen = time.time()
            return key_info.public_key
        return None
    
    def is_verified(self, onion_address: str) -> bool:
        """Check if we have a verified key for this onion"""
        return onion_address in self.verified_keys and \
               self.verified_keys[onion_address].verification_status == 'verified'
    
    def mark_pending(self, onion_address: str) -> None:
        """Mark that we're waiting for key exchange from this onion"""
        self.pending_verifications[onion_address] = time.time()
    
    def is_pending(self, onion_address: str) -> bool:
        """Check if we're waiting for a key exchange"""
        if onion_address in self.pending_verifications:
            # Timeout after 30 seconds
            if time.time() - self.pending_verifications[onion_address] > 30:
                del self.pending_verifications[onion_address]
                return False
            return True
        return False
    
    def clear_pending(self, onion_address: str) -> None:
        """Clear pending status"""
        if onion_address in self.pending_verifications:
            del self.pending_verifications[onion_address]

class YNXCrypto:
    @staticmethod
    def ynx_encode(hex_input: str) -> str:
        if not hex_input:
            raise ValueError("Empty input")
        
        try:
            num = int(hex_input, 16)
        except ValueError:
            raise ValueError(f"Invalid hex input: {hex_input}")
        
        if num == 0:
            return YNX_ALPHABET[0]
        
        result = ""
        while num > 0:
            num, remainder = divmod(num, 58)
            result = YNX_ALPHABET[remainder] + result
        
        # Handle leading zeros
        leading_zeros = 0
        for i in range(0, len(hex_input), 2):
            if i + 1 < len(hex_input) and hex_input[i:i+2] == '00':
                leading_zeros += 1
            else:
                break
        
        result = YNX_ALPHABET[0] * leading_zeros + result
        return result
    
    @staticmethod
    def ynx_decode(ynx_input: str) -> str:
        leading_zeros = 0
        for char in ynx_input:
            if char == YNX_ALPHABET[0]:
                leading_zeros += 1
            else:
                break
        
        num = 0
        for char in ynx_input:
            if char not in YNX_ALPHABET:
                raise ValueError(f"Invalid character: {char}")
            num = num * 58 + YNX_ALPHABET.index(char)
        
        if num == 0:
            hex_result = "00"
        else:
            hex_result = hex(num)[2:]
            if len(hex_result) % 2:
                hex_result = "0" + hex_result
        
        hex_result = "00" * leading_zeros + hex_result
        return hex_result
    
    @staticmethod
    def generate_ed25519_keypair() -> Tuple[str, str]:
        private_key_bytes = subprocess.check_output([
            "openssl", "genpkey", "-algorithm", "Ed25519", "-outform", "DER"
        ])
        private_key = private_key_bytes[-32:].hex()
        
        public_key_bytes = subprocess.check_output([
            "openssl", "pkey", "-inform", "DER", "-pubout", "-outform", "DER"
        ], input=private_key_bytes)
        public_key = public_key_bytes[-32:].hex()
        
        return private_key, public_key
    
    @staticmethod
    def sign_message(private_key: str, message: str) -> str:
        """Create Ed25519 signature for a message"""
        import tempfile
        timestamp = int(time.time())
        message_with_timestamp = f"{message}:{timestamp}"
        
        try:
            # Generate new Ed25519 key pair for signing
            result = subprocess.run([
                'openssl', 'genpkey', '-algorithm', 'Ed25519', '-out', '/dev/stdout'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return f"keygen_error:{timestamp}"
            
            sign_private_key = result.stdout
            
            # Get corresponding public key
            pub_result = subprocess.run([
                'openssl', 'pkey', '-in', '/dev/stdin', '-pubout', '-out', '/dev/stdout'
            ], input=sign_private_key, capture_output=True, text=True)
            
            if pub_result.returncode != 0:
                return f"pubkey_error:{timestamp}"
            
            sign_public_key = pub_result.stdout
            
            # Sign the message using temp files
            with tempfile.NamedTemporaryFile(mode='w') as priv_file, tempfile.NamedTemporaryFile(mode='w') as msg_file:
                priv_file.write(sign_private_key)
                priv_file.flush()
                
                msg_file.write(message_with_timestamp)
                msg_file.flush()
                
                sign_result = subprocess.run([
                    'openssl', 'pkeyutl', '-sign', '-inkey', priv_file.name, 
                    '-in', msg_file.name, '-out', '/dev/stdout'
                ], capture_output=True)
            
            if sign_result.returncode == 0:
                signature = sign_result.stdout.hex()
                # Return signature with the public key for verification
                return f"{signature}:{sign_public_key.replace(chr(10), '|')}:{timestamp}"
            else:
                return f"sign_error:{timestamp}"
        except Exception as e:
            return f"error:{timestamp}"
    
    @staticmethod
    def verify_signature(public_key: str, message: str, signature_with_data: str) -> bool:
        """Verify a message signature using the public key"""
        try:
            # New format: signature:public_key:timestamp
            parts = signature_with_data.split(':', 2)
            if len(parts) != 3:
                return False
            
            signature, sign_public_key_pem, timestamp_str = parts
            timestamp = int(timestamp_str)
            
            # Check timestamp is recent (within 5 minutes)
            if abs(time.time() - timestamp) > 300:
                return False
            
            # Reconstruct the signed data
            message_with_timestamp = f"{message}:{timestamp}"
            
            # Convert back from our encoding
            sign_public_key_pem = sign_public_key_pem.replace('|', chr(10))
            signature_bytes = binascii.unhexlify(signature)
            
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w') as pub_file, tempfile.NamedTemporaryFile(mode='w') as msg_file, tempfile.NamedTemporaryFile() as sig_file:
                pub_file.write(sign_public_key_pem)
                pub_file.flush()
                
                msg_file.write(message_with_timestamp)
                msg_file.flush()
                
                sig_file.write(signature_bytes)
                sig_file.flush()
                
                # Use openssl to verify with the signing public key
                result = subprocess.run([
                    'openssl', 'pkeyutl', '-verify', '-pubin', '-inkey', pub_file.name,
                    '-in', msg_file.name, '-sigfile', sig_file.name
                ], capture_output=True)
                
                debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
                if debug:
                    if result.returncode != 0:
                        os.environ['YNX_VERIFY_ERROR'] = result.stderr.decode()
                    else:
                        os.environ['YNX_VERIFY_ERROR'] = 'verified_ok'
                
                return result.returncode == 0
                
        except Exception as e:
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            if debug:
                os.environ['YNX_VERIFY_ERROR'] = str(e)
            return False
    
    @staticmethod
    def pubkey_to_ynx_address(public_key: str) -> str:
        return YNXCrypto.ynx_encode(public_key)
    
    @staticmethod
    def ynx_address_to_pubkey(ynx_address: str) -> str:
        return YNXCrypto.ynx_decode(ynx_address)
    
    @staticmethod
    def pubkey_to_onion_address(public_key: str) -> str:
        pubkey_bytes = binascii.unhexlify(public_key)
        
        version = b'\x03'
        checksum_input = b'.onion checksum' + pubkey_bytes + version
        checksum = hashlib.sha3_256(checksum_input).digest()[:2]
        
        onion_data = pubkey_bytes + checksum + version
        onion_address = base64.b32encode(onion_data).decode().lower().rstrip('=')
        
        return f"{onion_address}.onion"
    
    @staticmethod
    def extract_pubkey_from_onion(onion_address: str) -> str:
        """Extract public key from onion address"""
        onion_base = onion_address.replace('.onion', '')
        
        # Properly pad base32 string
        padding_needed = (8 - len(onion_base) % 8) % 8
        onion_padded = onion_base.upper() + '=' * padding_needed
        
        decoded = base64.b32decode(onion_padded)
        
        if len(decoded) == 35:
            return decoded[:32].hex()
        else:
            raise ValueError(f"Invalid onion address format: {onion_address}")

class YNXContacts:
    def __init__(self, ynx_dir: Path):
        self.contacts_file = ynx_dir / "contacts"
        self.ynx_dir = ynx_dir
        self.ynx_dir.mkdir(mode=0o700, exist_ok=True)
    
    def add_contact(self, ynx_address: str, name: str) -> str:
        if not ynx_address or not name:
            return "Usage: add <ynx_address> <name>"
        
        contacts = self._load_contacts()
        
        if name in contacts:
            contacts[name] = ynx_address
            self._save_contacts(contacts)
            return f"Updated contact: {name}"
        else:
            contacts[name] = ynx_address
            self._save_contacts(contacts)
            return f"Added contact: {name}"
    
    def list_contacts(self) -> str:
        contacts = self._load_contacts()
        
        if not contacts:
            return "No contacts found"
        
        result = ["Contacts:"]
        for name, address in contacts.items():
            result.append(f"  {name}: {address}")
        
        return "\n".join(result)
    
    def resolve_contact(self, name: str) -> Optional[str]:
        contacts = self._load_contacts()
        return contacts.get(name)
    
    def remove_contact(self, name: str) -> str:
        contacts = self._load_contacts()
        if name in contacts:
            del contacts[name]
            self._save_contacts(contacts)
            return f"Removed contact: {name}"
        else:
            return f"Contact not found: {name}"
    
    def _load_contacts(self) -> Dict[str, str]:
        if not self.contacts_file.exists():
            return {}
        
        contacts = {}
        try:
            for line in self.contacts_file.read_text().strip().split('\n'):
                if ':' in line:
                    name, address = line.split(':', 1)
                    contacts[name] = address
        except Exception:
            pass
        
        return contacts
    
    def _save_contacts(self, contacts: Dict[str, str]):
        content = "\n".join(f"{name}:{address}" for name, address in contacts.items())
        self.contacts_file.write_text(content)
        self.contacts_file.chmod(0o600)

class YNXTor:
    def __init__(self, profile_base: str, ynx_dir: Path):
        self.profile_base = profile_base
        self.ynx_dir = ynx_dir
        self.tor_data_dir = ynx_dir / f"tordata_{profile_base}"
        self.torrc_file = self.tor_data_dir / "torrc"
        self.service_dir = self.tor_data_dir / "ynx_service"
        self.ports_file = self.tor_data_dir / "ports"
        self.tor_pid_file = self.tor_data_dir / "tor.pid"
        
        self.socks_port = None
        self.control_port = None
        self.local_port = None
        
    def find_free_port(self, start_port: int) -> int:
        import socket
        for port in range(start_port, 65535):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No free port found")
    
    async def setup_tor_service(self) -> str:
        self.tor_data_dir.mkdir(mode=0o700, exist_ok=True)
        self.service_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Find free ports
        self.local_port = self.find_free_port(9165)
        self.socks_port = self.find_free_port(self.local_port + 1)
        self.control_port = self.find_free_port(self.socks_port + 1)
        
        # Save ports
        ports_content = f"socks:{self.socks_port}\ncontrol:{self.control_port}\nlocal:{self.local_port}\n"
        self.ports_file.write_text(ports_content)
        
        # Check if Tor is already running
        if self.tor_pid_file.exists():
            try:
                tor_pid = int(self.tor_pid_file.read_text().strip())
                os.kill(tor_pid, 0)
                
                hostname_file = self.service_dir / "hostname"
                if hostname_file.exists():
                    onion_address = hostname_file.read_text().strip()
                    print(f"Tor service already running (SOCKS: {self.socks_port}, Local: {self.local_port})")
                    return onion_address
                
            except (OSError, ValueError):
                self.tor_pid_file.unlink(missing_ok=True)
        
        # Create torrc
        torrc_content = f"""DataDirectory {self.tor_data_dir}
SocksPort {self.socks_port}
HiddenServiceDir {self.service_dir}
HiddenServicePort 2323 127.0.0.1:{self.local_port}
ControlPort {self.control_port}
CookieAuthentication 1
Log notice file {self.tor_data_dir}/tor.log
"""
        self.torrc_file.write_text(torrc_content)
        
        # Start Tor
        print(f"Starting Tor service on ports {self.socks_port} (SOCKS), {self.control_port} (control), {self.local_port} (local)...")
        
        process = await asyncio.create_subprocess_exec(
            'tor', '-f', str(self.torrc_file), '--PidFile', str(self.tor_pid_file), '--RunAsDaemon', '1',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Tor failed to start: {stderr.decode()}")
            raise RuntimeError(f"Tor startup failed: {stderr.decode()}")
        
        # Wait for SOCKS port to be ready
        print("Waiting for Tor to bootstrap...")
        import socket
        for i in range(30):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                result = sock.connect_ex(('127.0.0.1', self.socks_port))
                sock.close()
                if result == 0:
                    print(f"Tor SOCKS proxy ready on port {self.socks_port}")
                    break
            except:
                pass
            
            if i == 29:
                raise RuntimeError(f"Tor SOCKS port {self.socks_port} not responding after 30 seconds")
            
            await asyncio.sleep(1)
        
        # Wait for hidden service
        hostname_file = self.service_dir / "hostname"
        for _ in range(30):
            if hostname_file.exists():
                onion_address = hostname_file.read_text().strip()
                print(f"Tor hidden service ready: {onion_address}")
                return onion_address
            await asyncio.sleep(1)
        
        raise RuntimeError("Failed to start Tor hidden service")
    
    def stop_tor_service(self):
        if self.tor_pid_file.exists():
            try:
                tor_pid = int(self.tor_pid_file.read_text().strip())
                os.kill(tor_pid, signal.SIGTERM)
                self.tor_pid_file.unlink()
            except (OSError, ValueError):
                self.tor_pid_file.unlink(missing_ok=True)

class YNXProfileManager:
    def __init__(self, ynx_dir: Path):
        self.ynx_dir = ynx_dir
        self.ynx_dir.mkdir(mode=0o700, exist_ok=True)
    
    def create_profile(self, profile_name: str) -> Path:
        profile_file = self.ynx_dir / profile_name
        profile_file.write_text("pending")
        profile_file.chmod(0o600)
        return profile_file
    
    def load_profile(self, profile_name: str) -> Optional[YNXProfile]:
        profile_file = self.ynx_dir / profile_name
        
        if not profile_file.exists():
            return None
        
        content = profile_file.read_text().strip()
        if content == "pending":
            return None
        
        parts = content.split(":")
        if len(parts) >= 3:
            private_key, public_key, onion_address = parts[0], parts[1], parts[2]
            ynx_address = YNXCrypto.pubkey_to_ynx_address(public_key)
            
            return YNXProfile(
                name=profile_name,
                private_key=private_key,
                public_key=public_key,
                onion_address=onion_address,
                ynx_address=ynx_address
            )
        
        return None
    
    def finalize_profile_from_onion(self, profile_name: str, onion_address: str, tor: YNXTor) -> YNXProfile:
        profile_file = self.ynx_dir / profile_name
        
        # Extract public key from onion address
        onion_base = onion_address.replace('.onion', '')
        
        # Properly pad base32 string
        padding_needed = (8 - len(onion_base) % 8) % 8
        onion_padded = onion_base.upper() + '=' * padding_needed
        
        try:
            decoded = base64.b32decode(onion_padded)
        except Exception as e:
            raise RuntimeError(f"Failed to decode onion address: {e}")
        
        if len(decoded) == 35:
            public_key = decoded[:32].hex()
            
            # Read private key from Tor's files
            private_key_file = tor.service_dir / "hs_ed25519_secret_key"
            if private_key_file.exists():
                private_key = private_key_file.read_bytes()[-32:].hex()
                
                # Save to profile
                profile_content = f"{private_key}:{public_key}:{onion_address}"
                profile_file.write_text(profile_content)
                profile_file.chmod(0o600)
                
                ynx_address = YNXCrypto.pubkey_to_ynx_address(public_key)
                
                return YNXProfile(
                    name=profile_name,
                    private_key=private_key,
                    public_key=public_key,
                    onion_address=onion_address,
                    ynx_address=ynx_address
                )
        
        raise RuntimeError("Failed to extract keys from onion address")

class YNXMessageListener:
    def __init__(self, profile: YNXProfile, tor: YNXTor, ui, contacts, key_cache: YNXKeyCache):
        self.profile = profile
        self.tor = tor
        self.ui = ui
        self.contacts = contacts
        self.key_cache = key_cache
        self.server = None
        self.running = False
    
    async def start_listener(self):
        if self.running:
            return
        
        self.running = True
        try:
            self.server = await asyncio.start_server(
                self.handle_connection,
                '127.0.0.1',
                self.tor.local_port
            )
            
            print(f"Message listener started on port {self.tor.local_port}")
            
            # Start serving in background task
            asyncio.create_task(self.server.serve_forever())
            
        except Exception as e:
            self.ui.add_history_message(f"Failed to start listener: {e}")
    
    async def handle_connection(self, reader, writer):
        try:
            data = await reader.read(4096)  # Increased buffer for signatures
            line = data.decode().strip()
            
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            
            if debug:
                self.ui.add_history_message(f"Debug: Received: {line[:100]}...")
            
            # Handle key exchange request
            if line == "KEYEX_REQUEST":
                if debug:
                    self.ui.add_history_message("Debug: Received key exchange request")
                
                response = f"KEYEX_RESPONSE {self.profile.public_key}\n"
                writer.write(response.encode())
                await writer.drain()
                
                if debug:
                    self.ui.add_history_message("Debug: Sent public key")
            
            # Handle key exchange response
            elif line.startswith("KEYEX_RESPONSE "):
                public_key = line[15:].strip()
                
                if debug:
                    self.ui.add_history_message(f"Debug: Received public key response")
                
                # We don't know the onion address of the responder here
                # This will be handled by the messenger when it receives the response
                writer.write(b"ACK\n")
                await writer.drain()
            
            # Handle signed message
            elif line.startswith("SEND "):
                parts = line[5:].split(' ', 2)
                
                if len(parts) >= 3:
                    from_onion = parts[0]
                    
                    # Find where message starts and ends, and where signature starts
                    remaining = parts[1] + ' ' + parts[2]
                    
                    # Message is enclosed in quotes
                    if remaining.startswith('"'):
                        # Find the closing quote
                        end_quote_idx = remaining.find('"', 1)
                        if end_quote_idx > 0:
                            message = remaining[1:end_quote_idx]
                            
                            # Check if there's a signature after the message
                            signature_part = remaining[end_quote_idx + 1:].strip()
                            
                            if signature_part:
                                # We have a signature - verify it
                                cached_key = self.key_cache.get_public_key(from_onion)
                                
                                if not cached_key:
                                    # First message from this peer - request their key
                                    if debug:
                                        self.ui.add_history_message(f"Debug: No cached key for {from_onion}, requesting...")
                                    
                                    # Mark as pending and request key
                                    self.key_cache.mark_pending(from_onion)
                                    
                                    # Send key request in background
                                    async def verify_and_update(onion, msg_from_ynx):
                                        key = await self.request_public_key(onion)
                                        if key:
                                            self.ui.add_history_message(f"Identity verified for {msg_from_ynx}")
                                        else:
                                            self.ui.add_history_message(f"⚠ Could not verify identity for {msg_from_ynx}")
                                    
                                    from_ynx = self.get_display_name_from_onion(from_onion)
                                    asyncio.create_task(verify_and_update(from_onion, from_ynx))
                                    
                                    # Still display message but mark as unverified
                                    self.ui.add_history_message(f"{from_ynx} [unverified]> {message}")
                                else:
                                    # Verify signature
                                    if debug:
                                        self.ui.add_history_message(f"Debug: Verifying sig '{signature_part[:50]}...' with key '{cached_key[:16]}...' for msg '{message}'")
                                    
                                    verify_result = YNXCrypto.verify_signature(cached_key, message, signature_part)
                                    
                                    if verify_result:
                                        from_ynx = self.get_display_name_from_onion(from_onion)
                                        self.ui.add_history_message(f"{from_ynx}> {message}")
                                        if debug:
                                            self.ui.add_history_message("Debug: Message verified")
                                    else:
                                        from_ynx = self.get_display_name_from_onion(from_onion)
                                        self.ui.add_history_message(f"{from_ynx} ⚠> {message}")
                                        self.ui.add_history_message("⚠ Warning: Message signature invalid!")
                                        if debug:
                                            self.ui.add_history_message("Debug: Signature verification failed")
                                            # Show OpenSSL error if available
                                            verify_error = os.environ.get('YNX_VERIFY_ERROR', '')
                                            if verify_error and verify_error != 'success':
                                                self.ui.add_history_message(f"Debug: OpenSSL error: {verify_error}")
                                                os.environ['YNX_VERIFY_ERROR'] = ''  # Clear it
                            else:
                                # No signature - old client or modified protocol
                                from_ynx = self.get_display_name_from_onion(from_onion)
                                self.ui.add_history_message(f"{from_ynx} [unsigned]> {message}")
                                if debug:
                                    self.ui.add_history_message("Debug: Message has no signature")
                            
                            writer.write(b"ACK\n")
                        else:
                            writer.write(b"INVALID_FORMAT\n")
                            if debug:
                                self.ui.add_history_message("Debug: Message format invalid - no closing quote")
                    else:
                        writer.write(b"INVALID_FORMAT\n")
                        if debug:
                            self.ui.add_history_message("Debug: Message format invalid - no opening quote")
                else:
                    writer.write(b"INVALID_FORMAT\n")
                    if debug:
                        self.ui.add_history_message("Debug: Wrong number of parts in SEND command")
            
            else:
                writer.write(b"UNKNOWN_COMMAND\n")
                if debug:
                    self.ui.add_history_message(f"Debug: Unknown command: {line[:50]}")
            
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            if debug:
                self.ui.add_history_message(f"Debug: Connection error: {e}")
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
    
    async def request_public_key(self, target_onion: str) -> Optional[str]:
        """Request public key from a peer"""
        try:
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            
            # Use Python native socket
            loop = asyncio.get_event_loop()
            response, debug_msgs = await loop.run_in_executor(
                None, socks5_send_receive, target_onion, "KEYEX_REQUEST", self.tor.socks_port, 15.0
            )
            
            if debug:
                for msg in debug_msgs:
                    self.ui.add_history_message(f"Debug SOCKS5: {msg}")
            
            if debug:
                self.ui.add_history_message(f"Debug: Got key response: {response[:100]}")
            
            if response.startswith("KEYEX_RESPONSE "):
                public_key = response[15:].strip()
                
                # Verify that the public key matches the onion address
                expected_onion = YNXCrypto.pubkey_to_onion_address(public_key)
                if expected_onion == target_onion:
                    self.key_cache.add_verified_key(target_onion, public_key)
                    self.key_cache.clear_pending(target_onion)
                    
                    if debug:
                        self.ui.add_history_message(f"Debug: Successfully verified key for {target_onion}")
                    
                    return public_key
                else:
                    if debug:
                        self.ui.add_history_message(f"Debug: Key mismatch for {target_onion}")
                    return None
            
            return None
            
        except asyncio.TimeoutError:
            self.key_cache.clear_pending(target_onion)
            if debug:
                self.ui.add_history_message(f"Debug: Key exchange timeout with {target_onion}")
            return None
        except Exception as e:
            self.key_cache.clear_pending(target_onion)
            if debug:
                self.ui.add_history_message(f"Debug: Key exchange error: {e}")
            return None
    
    def get_display_name_from_onion(self, from_onion: str) -> str:
        """Get display name for message sender from onion address"""
        try:
            from_pubkey = YNXCrypto.extract_pubkey_from_onion(from_onion)
            from_ynx = YNXCrypto.pubkey_to_ynx_address(from_pubkey)
            
            # Check if sender is in contacts
            contacts = self.contacts._load_contacts()
            for name, ynx_addr in contacts.items():
                if ynx_addr == from_ynx:
                    return name
            
            # If not in contacts but is current chat target
            if self.ui.current_chat_target == from_pubkey:
                return "they"
            
            return from_ynx
        except:
            return from_onion[:16] + "..."
    
    def stop_listener(self):
        self.running = False
        if self.server:
            self.server.close()

class YNXMessenger:
    def __init__(self, profile: YNXProfile, tor: YNXTor, ui, key_cache: YNXKeyCache):
        self.profile = profile
        self.tor = tor
        self.ui = ui
        self.key_cache = key_cache
    
    async def send_message(self, target_pubkey: str, message: str) -> bool:
        try:
            target_onion = YNXCrypto.pubkey_to_onion_address(target_pubkey)
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            
            # Check if we have the target's public key cached
            if not self.key_cache.is_verified(target_onion):
                if not self.key_cache.is_pending(target_onion):
                    if debug:
                        self.ui.add_history_message(f"Debug: Requesting public key from {target_onion}")
                    
                    # Request public key first
                    self.key_cache.mark_pending(target_onion)
                    public_key = await self.request_and_verify_key(target_onion)
                    
                    if not public_key:
                        self.ui.add_history_message("Failed to verify recipient's identity")
                        return False
            
            # Sign the message
            signature = YNXCrypto.sign_message(self.profile.private_key, message)
            
            # Create signed message command
            send_command = f'SEND {self.profile.onion_address} "{message}" {signature}\n'
            
            if debug:
                self.ui.add_history_message(f"Debug: Sending signed message to {target_onion}")
            
            # Use Python native socket
            loop = asyncio.get_event_loop()
            response, debug_msgs = await loop.run_in_executor(
                None, socks5_send_receive, target_onion, send_command.strip(), self.tor.socks_port, 15.0
            )
            
            if debug:
                for msg in debug_msgs:
                    self.ui.add_history_message(f"Debug SOCKS5: {msg}")
            
            if debug:
                self.ui.add_history_message(f"Debug: Response: {response}")
            
            if response.strip() == "ACK":
                return True
            else:
                self.ui.add_history_message("Failed to deliver message (peer offline)")
                return False
            
        except asyncio.TimeoutError:
            self.ui.add_history_message("Message timeout (peer offline)")
            return False
        except Exception as e:
            self.ui.add_history_message(f"Send error: {e}")
            return False
    
    async def request_and_verify_key(self, target_onion: str) -> Optional[str]:
        """Request and verify public key from target"""
        try:
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            
            # Use Python native socket
            loop = asyncio.get_event_loop()
            response, debug_msgs = await loop.run_in_executor(
                None, socks5_send_receive, target_onion, "KEYEX_REQUEST", self.tor.socks_port, 20.0
            )
            
            if debug:
                for msg in debug_msgs:
                    self.ui.add_history_message(f"Debug SOCKS5: {msg}")
            
            if debug:
                self.ui.add_history_message(f"Debug: Key exchange response: {response[:100]}")
            
            if response.startswith("KEYEX_RESPONSE "):
                public_key = response[15:].strip()
                
                # Verify the public key matches the onion address
                expected_onion = YNXCrypto.pubkey_to_onion_address(public_key)
                if expected_onion == target_onion:
                    self.key_cache.add_verified_key(target_onion, public_key)
                    self.key_cache.clear_pending(target_onion)
                    if debug:
                        self.ui.add_history_message(f"Debug: Key verified successfully for {target_onion}")
                    return public_key
                else:
                    if debug:
                        self.ui.add_history_message(f"Debug: Expected {expected_onion}, got {target_onion}")
                    self.ui.add_history_message("⚠ Key verification failed - public key doesn't match onion address")
                    return None
            
            if debug:
                self.ui.add_history_message(f"Debug: No KEYEX_RESPONSE in: {response[:100]}")
            return None
            
        except asyncio.TimeoutError:
            self.key_cache.clear_pending(target_onion)
            if debug:
                self.ui.add_history_message(f"Debug: Key exchange timeout with {target_onion}")
            return None
        except Exception as e:
            self.key_cache.clear_pending(target_onion)
            debug = os.environ.get('YNX_DEBUG', '').lower() == 'true'
            if debug:
                self.ui.add_history_message(f"Debug: Key exchange error: {e}")
            else:
                self.ui.add_history_message(f"Key exchange error: {e}")
            return None

class YNXCursesUI:
    def __init__(self, profile_base: str):
        self.profile_base = profile_base
        self.history_messages = []
        self.input_buffer = ""
        self.current_chat_target = None
        self.stdscr = None
        self.history_win = None
        self.input_win = None
        self.running = True
        self.is_typing = False
        
        # Setup history file with random suffix
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        self.history_file = Path(f"/tmp/ynx_history_{profile_base}_{random_suffix}.log")
        self.history_file.write_text("")
        self.history_file.chmod(0o600)
    
    def add_history_message(self, message: str):
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        # Split multi-line messages
        lines = message.split('\n')
        if len(lines) > 1:
            # First line with timestamp
            self.history_messages.append(f"[{timestamp}] {lines[0]}")
            # Additional lines without timestamp
            for line in lines[1:]:
                self.history_messages.append(f"            {line}")
        else:
            self.history_messages.append(formatted_message)
        
        # Also write to file
        with open(self.history_file, 'a') as f:
            f.write(formatted_message + "\n")
        
        if self.history_win:
            self.refresh_history()
            # Always keep cursor in input window
            self.focus_input()
    
    def refresh_history(self):
        if not self.history_win:
            return
            
        self.history_win.clear()
        height, width = self.history_win.getmaxyx()
        
        # Wrap long lines and fit to window
        wrapped_lines = []
        for message in self.history_messages:
            if len(message) > width - 1:
                # Wrap long lines
                while len(message) > width - 1:
                    wrapped_lines.append(message[:width-1])
                    message = "    " + message[width-1:]  # Indent continuation
                wrapped_lines.append(message)
            else:
                wrapped_lines.append(message)
        
        # Show last lines that fit in window
        start_idx = max(0, len(wrapped_lines) - height + 1)
        for i, line in enumerate(wrapped_lines[start_idx:]):
            if i < height - 1:
                try:
                    self.history_win.addstr(i, 0, line)
                except curses.error:
                    pass
        
        self.history_win.refresh()
    
    def refresh_input(self):
        if not self.input_win:
            return
            
        self.input_win.clear()
        height, width = self.input_win.getmaxyx()
        
        if self.current_chat_target:
            prompt = "you> "
        else:
            prompt = "ynx> "
        
        display_text = prompt + self.input_buffer
        
        # Truncate if too long for window
        if len(display_text) > width - 1:
            display_text = display_text[:width-1]
        
        try:
            self.input_win.addstr(0, 0, display_text)
            # Position cursor at end of input
            cursor_pos = min(len(display_text), width - 1)
            self.input_win.move(0, cursor_pos)
        except curses.error:
            pass
        
        self.input_win.refresh()
        
        # Move cursor to correct position in input window
        try:
            cursor_pos = min(len(display_text), width - 1)
            self.input_win.move(0, cursor_pos)
        except curses.error:
            pass
    
    def init_windows(self, stdscr):
        self.stdscr = stdscr
        
        # Clear screen and show cursor
        stdscr.clear()
        curses.curs_set(1)
        
        # Get terminal size
        height, width = stdscr.getmaxyx()
        
        # Create windows: history takes most space, input takes 3 lines at bottom
        history_height = height - 4  # Leave more space
        separator_line = history_height
        input_start = history_height + 1
        
        self.history_win = curses.newwin(history_height, width, 0, 0)
        self.input_win = curses.newwin(3, width, input_start, 0)
        
        # Enable scrolling for history window
        self.history_win.scrollok(True)
        
        # Draw separator line
        try:
            stdscr.addstr(separator_line, 0, "─" * (width - 1))
        except curses.error:
            pass
        
        stdscr.refresh()
        self.refresh_history()
        self.refresh_input()
    
    def handle_input(self, ch):
        self.is_typing = True
        
        if ch == ord('\n') or ch == 10:  # Enter
            if self.input_buffer.strip():
                self.is_typing = False
                return self.input_buffer.strip()
            return None
        elif ch == 127 or ch == curses.KEY_BACKSPACE:  # Backspace
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                self.update_input_display()
        elif ch == 3:  # Ctrl+C
            self.running = False
            self.is_typing = False
            return "/exit"
        elif 32 <= ch <= 126:  # Printable characters
            self.input_buffer += chr(ch)
            self.update_input_display()
        
        return None
    
    def update_input_display(self):
        """Update only the input line without full refresh"""
        if not self.input_win:
            return
        
        height, width = self.input_win.getmaxyx()
        
        if self.current_chat_target:
            prompt = "you> "
        else:
            prompt = "ynx> "
        
        display_text = prompt + self.input_buffer
        
        # Clear line and redraw
        self.input_win.move(0, 0)
        self.input_win.clrtoeol()
        
        # Truncate if too long
        if len(display_text) > width - 1:
            display_text = display_text[:width-1]
        
        try:
            self.input_win.addstr(0, 0, display_text)
            cursor_pos = min(len(prompt) + len(self.input_buffer), width - 1)
            self.input_win.move(0, cursor_pos)
            self.input_win.refresh()
        except curses.error:
            pass
    
    def focus_input(self):
        """Force cursor back to input window"""
        if not self.input_win:
            return
        
        try:
            # Calculate cursor position
            if self.current_chat_target:
                prompt = "you> "
            else:
                prompt = "ynx> "
            
            cursor_pos = len(prompt) + len(self.input_buffer)
            height, width = self.input_win.getmaxyx()
            cursor_pos = min(cursor_pos, width - 1)
            
            # Force cursor to input window
            self.input_win.move(0, cursor_pos)
            self.input_win.refresh()
            
            # Also tell main screen to move cursor there
            if self.stdscr:
                input_y, input_x = self.input_win.getbegyx()
                self.stdscr.move(input_y, cursor_pos)
                curses.curs_set(1)  # Make cursor visible
                self.stdscr.refresh()
        except curses.error:
            pass
    
    def handle_resize(self, stdscr):
        """Handle terminal resize event"""
        try:
            # Clear and refresh the main screen
            stdscr.clear()
            stdscr.refresh()
            
            # Get new terminal size
            height, width = stdscr.getmaxyx()
            
            # Destroy old windows
            if self.history_win:
                del self.history_win
            if self.input_win:
                del self.input_win
            
            # Recreate windows with new size
            history_height = height - 4
            separator_line = history_height
            input_start = history_height + 1
            
            self.history_win = curses.newwin(history_height, width, 0, 0)
            self.input_win = curses.newwin(3, width, input_start, 0)
            
            # Enable scrolling for history window
            self.history_win.scrollok(True)
            
            # Redraw separator line
            try:
                stdscr.addstr(separator_line, 0, "─" * (width - 1))
            except curses.error:
                pass
            
            stdscr.refresh()
            
            # Refresh both windows with content
            self.refresh_history()
            self.refresh_input()
            self.focus_input()
            
        except curses.error:
            pass

class YNXClient:
    def __init__(self, profile_name: str = "default.ynx", ynx_dir: Optional[Path] = None):
        self.profile_name = profile_name
        self.profile_base = profile_name.replace('.ynx', '')
        self.ynx_dir = ynx_dir or (Path.home() / ".ynx")
        self.is_temp_profile = profile_name.startswith('temp_')
        
        # Initialize components
        self.profile_manager = YNXProfileManager(self.ynx_dir)
        self.contacts = YNXContacts(self.ynx_dir)
        self.tor = YNXTor(self.profile_base, self.ynx_dir)
        self.ui = YNXCursesUI(self.profile_base)
        self.key_cache = YNXKeyCache()  # New key cache
        
        # Runtime state
        self.profile = None
        self.listener = None
        self.messenger = None
        
    async def initialize(self):
        
        # Check dependencies
        for cmd in ['tor', 'openssl']:
            if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
                print(f"Error: Required command '{cmd}' not found")
                print(f"Please install: {cmd}")
                return False
        
        # Load or create profile
        profile_file = self.ynx_dir / self.profile_name
        if profile_file.exists():
            print(f"Loading profile: {self.profile_name}")
            self.profile = self.profile_manager.load_profile(self.profile_name)
        else:
            print(f"Creating new profile: {self.profile_name}")
            self.profile_manager.create_profile(self.profile_name)
        
        # Setup Tor service
        onion_address = await self.tor.setup_tor_service()
        
        # Finalize profile with Tor-generated keys
        if onion_address and not self.profile:
            self.profile = self.profile_manager.finalize_profile_from_onion(
                self.profile_name, onion_address, self.tor
            )
        
        # Initialize messenger and listener with key cache
        if self.profile:
            self.messenger = YNXMessenger(self.profile, self.tor, self.ui, self.key_cache)
            self.listener = YNXMessageListener(self.profile, self.tor, self.ui, self.contacts, self.key_cache)
            
            await self.listener.start_listener()
            
            print(f"\nReady! Your YNX address: {self.profile.ynx_address}")
            print("Starting interface...")
            await asyncio.sleep(1)
            
            return True
        
        return False
    
    def run_ui(self, stdscr):
        self.ui.init_windows(stdscr)
        
        # Add initial messages
        self.ui.add_history_message(f"YNX Ready! Address: {self.profile.ynx_address}")
        self.ui.add_history_message("Type '/help' for commands")
        
        # Make sure cursor starts in input area
        self.ui.refresh_input()
        
        # Main input loop - use synchronous version for curses
        while self.ui.running:
            try:
                ch = stdscr.getch()
                
                # Handle terminal resize
                if ch == curses.KEY_RESIZE:
                    self.ui.handle_resize(stdscr)
                    continue
                
                command = self.ui.handle_input(ch)
                
                if command:
                    # Log command to history
                    if self.ui.current_chat_target:
                        self.ui.add_history_message(f"you> {command}")
                    else:
                        self.ui.add_history_message(f"ynx> {command}")
                    
                    # Clear input buffer
                    self.ui.input_buffer = ""
                    self.ui.refresh_input()
                    
                    # Process command synchronously for curses compatibility
                    self.process_command_sync(command)
                    
                    # Ensure cursor stays in input area
                    self.ui.refresh_input()
                    
            except KeyboardInterrupt:
                self.ui.running = False
                break
        
        self.cleanup()
    
    def process_command_sync(self, user_input: str):
        """Synchronous version for curses compatibility"""
        parts = user_input.split(' ', 1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            self.ui.add_history_message("Commands:")
            self.ui.add_history_message("  /chat <ynx_address|contact_name>  - Start chat session")
            self.ui.add_history_message("  /add <ynx_address> <name>         - Add contact")
            self.ui.add_history_message("  /remove <name>                    - Remove contact")
            self.ui.add_history_message("  /contacts                         - List contacts")
            self.ui.add_history_message("  /status                          - Show current chat session")
            self.ui.add_history_message("  /keys                            - Show cached verification keys")
            self.ui.add_history_message("  /whoami                          - Show your address/onion")
            if self.ui.current_chat_target:
                self.ui.add_history_message("  /exit                           - Exit chat")
            else:
                self.ui.add_history_message("  /exit                           - Quit program")
            
        elif cmd == "/chat":
            if not args:
                self.ui.add_history_message("Usage: /chat <ynx_address|contact_name>")
                return
            
            target_pubkey = self.resolve_target_sync(args)
            if target_pubkey:
                self.ui.current_chat_target = target_pubkey
                target_ynx = YNXCrypto.pubkey_to_ynx_address(target_pubkey)
                
                # Get display name for chat target
                display_name = self.get_chat_display_name(target_ynx)
                self.ui.add_history_message(f"Starting chat with {display_name}")
                
                # Check if we have verified key
                target_onion = YNXCrypto.pubkey_to_onion_address(target_pubkey)
                if self.key_cache.is_verified(target_onion):
                    self.ui.add_history_message("Identity previously verified")
            
        elif cmd == "/add":
            if not args:
                self.ui.add_history_message("Usage: /add <ynx_address> <name>")
                return
            
            addr_parts = args.split(' ', 1)
            if len(addr_parts) < 2:
                self.ui.add_history_message("Usage: /add <ynx_address> <name>")
                return
            
            addr, name = addr_parts
            result = self.contacts.add_contact(addr, name)
            self.ui.add_history_message(result)
            
        elif cmd == "/remove":
            if not args:
                self.ui.add_history_message("Usage: /remove <name>")
                return
            
            result = self.contacts.remove_contact(args)
            self.ui.add_history_message(result)
            
        elif cmd == "/contacts":
            result = self.contacts.list_contacts()
            self.ui.add_history_message(result)
            
        elif cmd == "/keys":
            if not self.key_cache.verified_keys:
                self.ui.add_history_message("No verified keys in cache")
            else:
                self.ui.add_history_message("Cached verification keys:")
                for onion, key_info in self.key_cache.verified_keys.items():
                    ynx = YNXCrypto.pubkey_to_ynx_address(key_info.public_key)
                    age = int(time.time() - key_info.verified_at)
                    self.ui.add_history_message(f"  {ynx[:16]}... (verified {age}s ago)")
            
        elif cmd == "/status":
            if self.ui.current_chat_target:
                target_ynx = YNXCrypto.pubkey_to_ynx_address(self.ui.current_chat_target)
                display_name = self.get_chat_display_name(target_ynx)
                self.ui.add_history_message(f"Currently chatting with: {display_name}")
                self.ui.add_history_message(f"YNX Address: {target_ynx}")
                
                # Check verification status
                target_onion = YNXCrypto.pubkey_to_onion_address(self.ui.current_chat_target)
                if self.key_cache.is_verified(target_onion):
                    self.ui.add_history_message("Identity verified")
                else:
                    self.ui.add_history_message("⚠ Identity not yet verified")
            else:
                self.ui.add_history_message("Not in a chat session")
            
        elif cmd == "/whoami":
            self.ui.add_history_message(f"YNX Address: {self.profile.ynx_address}")
            self.ui.add_history_message(f"Onion Address: {self.profile.onion_address}")
            
        elif cmd == "/exit":
            if self.ui.current_chat_target:
                # Exit chat, return to main
                self.ui.current_chat_target = None
                self.ui.add_history_message("Exited chat")
            else:
                # Exit program
                self.ui.add_history_message("Goodbye!")
                self.ui.running = False
            
        else:
            if self.ui.current_chat_target:
                # In chat mode - only send if not a slash command
                if user_input.startswith('/'):
                    self.ui.add_history_message(f"Unknown command: {cmd}")
                else:
                    # Send message (no slash = regular message)
                    def send_async():
                        asyncio.run_coroutine_threadsafe(
                            self.messenger.send_message(self.ui.current_chat_target, user_input),
                            self.event_loop
                        )
                    
                    thread = threading.Thread(target=send_async)
                    thread.daemon = True
                    thread.start()
            else:
                if user_input.startswith('/'):
                    self.ui.add_history_message(f"Unknown command: {cmd}")
                else:
                    self.ui.add_history_message("Use /help for commands")
    
    def get_chat_display_name(self, ynx_address: str) -> str:
        """Get display name for chat target"""
        contacts = self.contacts._load_contacts()
        for name, addr in contacts.items():
            if addr == ynx_address:
                return name
        return ynx_address
    
    def resolve_target_sync(self, target: str) -> Optional[str]:
        """Synchronous version for curses compatibility"""
        if len(target) == 64:
            return target
        elif 20 < len(target) < 60:
            try:
                return YNXCrypto.ynx_address_to_pubkey(target)
            except ValueError:
                self.ui.add_history_message(f"Invalid YNX address: {target}")
                return None
        else:
            ynx_addr = self.contacts.resolve_contact(target)
            if ynx_addr:
                try:
                    return YNXCrypto.ynx_address_to_pubkey(ynx_addr)
                except ValueError:
                    self.ui.add_history_message(f"Invalid YNX address for contact {target}")
                    return None
            else:
                self.ui.add_history_message(f"Contact not found: {target}")
                return None
    
    def cleanup(self):
        if self.listener:
            self.listener.stop_listener()
        if self.tor:
            self.tor.stop_tor_service()
        
        # Clean up history file for all profiles
        try:
            if hasattr(self.ui, 'history_file') and self.ui.history_file.exists():
                self.ui.history_file.unlink()
        except Exception:
            pass  # Silent cleanup - don't error on exit
        
        # Delete temp profile files if this is a temp profile
        if self.is_temp_profile:
            import shutil
            try:
                # Delete the entire temp ynx directory
                if self.ynx_dir.exists():
                    shutil.rmtree(self.ynx_dir)
            except Exception:
                pass  # Silent cleanup - don't error on exit

def main():
    import random
    import string
    
    profile_name = os.environ.get('YNX_PROFILE', 'default')
    
    # Handle temp profile - create random profile in /tmp
    if profile_name == 'temp':
        # Use PID and random suffix for uniqueness
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        pid = os.getpid()
        profile_name = f"temp_{pid}_{random_suffix}.ynx"
        # Override ynx_dir to use /tmp for temp profiles
        temp_ynx_dir = Path(f"/tmp/ynx_temp_{pid}_{random_suffix}")
        client = YNXClient(profile_name, temp_ynx_dir)
    else:
        # Add .ynx extension if not present
        if not profile_name.endswith('.ynx'):
            profile_name += '.ynx'
        client = YNXClient(profile_name)
    
    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize in event loop
    if not loop.run_until_complete(client.initialize()):
        return
    
    # Handle cleanup on exit
    def signal_handler(signum, frame):
        client.cleanup()
        loop.call_soon_threadsafe(loop.stop)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start event loop in background thread
    import threading
    def run_event_loop():
        try:
            loop.run_forever()
        except:
            pass
    
    loop_thread = threading.Thread(target=run_event_loop, daemon=True)
    loop_thread.start()
    
    # Store loop reference for async operations
    client.event_loop = loop
    
    try:
        # Run curses UI in main thread
        curses.wrapper(client.run_ui)
    finally:
        # Stop event loop gracefully
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        client.cleanup()
        
        # Wait for loop thread to finish
        if loop_thread.is_alive():
            loop_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT_END
    
    # Execute the script with terminal input and clean up on exit
    trap "rm -f '$TEMP_SCRIPT'" EXIT
    exec python3 "$TEMP_SCRIPT" < /dev/tty
}

# Run main function
main "$@"