"""
Password authentication for emergency shortcuts.

Handles password validation, failed attempt tracking, and account locking
for emergency shortcuts (Ctrl+E, Ctrl+F).
"""

import time
import logging
from typing import Dict


logger = logging.getLogger(__name__)


class SilverBulletError(Exception):
    """Base exception for Silver Bullet application."""

    pass


class PasswordAuthenticator:
    """Authenticator for emergency shortcut passwords."""

    def __init__(self, max_failed_attempts: int = 5, lockout_duration_minutes: int = 30):
        """Initialize password authenticator.

        Args:
            max_failed_attempts: Maximum failed attempts before lockout (default: 5)
            lockout_duration_minutes: How long to lock account after max failures (default: 30)
        """
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_seconds = lockout_duration_minutes * 60
        self._failed_attempts: Dict[str, int] = {}
        self._lockout_until: Dict[str, float] = {}

    def validate_password(self, shortcut_action: str, password: str) -> bool:
        """Validate password for a shortcut action.

        Args:
            shortcut_action: The shortcut action (e.g., 'emergency_stop', 'manual_flatten')
            password: The password to validate

        Returns:
            True if password is correct, False otherwise
        """
        # Check if account is locked
        if self.is_locked(shortcut_action):
            logger.warning(f"Account locked for action: {shortcut_action}")
            return False

        # Get correct password from config
        correct_password = self._get_correct_password(shortcut_action)

        if password == correct_password:
            # Reset failed attempts on success
            self._reset_failed_attempts(shortcut_action)
            logger.info(f"Password authentication successful for action: {shortcut_action}")
            return True
        else:
            # Increment failed attempts
            self._increment_failed_attempts(shortcut_action)
            logger.warning(f"Password authentication failed for action: {shortcut_action}")

            # Check if should lock account
            if self._failed_attempts[shortcut_action] >= self.max_failed_attempts:
                self._lock_account(shortcut_action)
                logger.error(f"Account locked for action: {shortcut_action} after {self.max_failed_attempts} failed attempts")

            return False

    def is_locked(self, shortcut_action: str) -> bool:
        """Check if account is currently locked.

        Args:
            shortcut_action: The shortcut action to check

        Returns:
            True if locked, False otherwise
        """
        if shortcut_action not in self._lockout_until:
            return False

        # Check if lockout has expired
        if time.time() > self._lockout_until[shortcut_action]:
            # Lockout expired, clear it
            del self._lockout_until[shortcut_action]
            self._reset_failed_attempts(shortcut_action)
            return False

        return True

    def get_failed_attempts(self, shortcut_action: str) -> int:
        """Get number of failed attempts for an action.

        Args:
            shortcut_action: The shortcut action

        Returns:
            Number of failed attempts
        """
        return self._failed_attempts.get(shortcut_action, 0)

    def get_lockout_remaining(self, shortcut_action: str) -> int:
        """Get remaining lockout time in seconds.

        Args:
            shortcut_action: The shortcut action

        Returns:
            Remaining seconds, or 0 if not locked
        """
        if shortcut_action not in self._lockout_until:
            return 0

        remaining = int(self._lockout_until[shortcut_action] - time.time())
        return max(0, remaining)

    def _get_correct_password(self, shortcut_action: str) -> str:
        """Get correct password for shortcut action from config.

        Args:
            shortcut_action: The shortcut action

        Returns:
            The correct password
        """
        # For testing and initial implementation, use default passwords
        # In production, this would load from config file
        test_passwords = {
            "emergency_stop": "test_password_123",
            "manual_flatten": "test_password_123"
        }

        return test_passwords.get(shortcut_action, "test_password_123")

    def _increment_failed_attempts(self, shortcut_action: str) -> None:
        """Increment failed attempt counter."""
        self._failed_attempts[shortcut_action] = self._failed_attempts.get(shortcut_action, 0) + 1

    def _reset_failed_attempts(self, shortcut_action: str) -> None:
        """Reset failed attempt counter."""
        if shortcut_action in self._failed_attempts:
            del self._failed_attempts[shortcut_action]

    def _lock_account(self, shortcut_action: str) -> None:
        """Lock account for specified duration."""
        self._lockout_until[shortcut_action] = time.time() + self.lockout_duration_seconds


class ShortcutAuthenticationError(SilverBulletError):
    """Raised when shortcut authentication fails."""

    pass


class AccountLockedError(SilverBulletError):
    """Raised when account is locked due to too many failed attempts."""

    def __init__(self, shortcut_action: str, remaining_seconds: int):
        """Initialize error.

        Args:
            shortcut_action: The locked shortcut action
            remaining_seconds: Remaining lockout time in seconds
        """
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        super().__init__(
            f"Account locked for action '{shortcut_action}'. "
            f"Try again in {minutes}m {seconds}s."
        )
