import os
import hashlib
import binascii

def hash_pass(password):
    """Hash a password for storing."""

    salt = os.urandom(32)  # Generate a random salt
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return salt + pwdhash.decode('ascii')  # Decode hexlify result to string


def verify_pass(provided_password, stored_password):
    """Verify a stored password against one provided by user"""

    stored_salt = stored_password[:32]
    stored_password = stored_password[32:]

    pwdhash = hashlib.pbkdf2_hmac('sha256',
                                  provided_password.encode('utf-8'),  # Encode password to bytes
                                  stored_salt,
                                  100000)
    
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')

    return pwdhash == stored_password
