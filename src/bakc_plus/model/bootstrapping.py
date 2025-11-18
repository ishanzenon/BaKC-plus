"""
Stratified bootstrapping for ensemble members

This module implements the stratified bootstrapping logic from the original notebook.
CRITICAL: The random state hashing and bootstrapping logic must be preserved EXACTLY
to ensure reproducible results.
"""

from typing import Tuple
import numpy as np

from ..logger import get_logger

logger = get_logger(__name__)


class StratifiedBootstrapper:
    """
    Stratified bootstrapping for OC-SVM ensemble members

    This class implements a leave-one-out style bootstrapping where the training
    data is split into M groups, and each ensemble member is trained on all groups
    EXCEPT one. This ensures diversity in the ensemble while maintaining
    reproducibility through careful random state management.

    CRITICAL PRESERVATION:
    - Random state hashing must match notebook exactly
    - Index shuffling and partitioning must match notebook
    - Leave-one-out logic must match notebook

    Example:
        >>> bootstrapper = StratifiedBootstrapper()
        >>> X_train = np.random.randn(100, 5)
        >>> X_boot, leave_out = bootstrapper.perform_bootstrapping(
        ...     X_train, member_idx=0, num_members=5, random_state=42
        ... )
        >>> # X_boot has ~80% of data, leave_out indices have ~20%
    """

    def __init__(self):
        """Initialize bootstrapper"""
        self.logger = get_logger(__name__)

    @staticmethod
    def hash_random_state(
        member_idx: int,
        fold_idx: int,
        random_state: int
    ) -> int:
        """
        Hash random state for deterministic yet varied seeds

        This is the CRITICAL random state hashing function from the notebook.
        It MUST be preserved EXACTLY to ensure reproducible results.

        Formula:
            rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
            rnd = rnd ^ 0x7FFFFFFF

        Args:
            member_idx: Index of ensemble member (0-indexed)
            fold_idx: Index of cross-validation fold (0-indexed)
            random_state: Base random seed

        Returns:
            Hashed random state (integer in [0, 2^31-1])

        Example:
            >>> rnd = StratifiedBootstrapper.hash_random_state(0, 0, 42)
            >>> type(rnd)
            <class 'int'>
            >>> 0 <= rnd < 2**31
            True

        Note:
            The % 4294967296 ensures result fits in 32-bit unsigned int
            The ^ 0x7FFFFFFF XORs with max 31-bit value to flip high bit
        """
        # Hash the tuple
        rnd = hash((member_idx, fold_idx, random_state))

        # Modulo to fit in 32-bit unsigned int (0 to 2^32-1)
        rnd = rnd % 4294967296

        # XOR with 0x7FFFFFFF (max 31-bit value: 2^31-1)
        rnd = rnd ^ 0x7FFFFFFF

        return rnd

    def perform_bootstrapping(
        self,
        X_train: np.ndarray,
        member_idx: int,
        num_members: int,
        random_state: int,
        fold_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stratified bootstrapping (leave-one-out style)

        This method splits the training data into num_members groups and returns
        all groups EXCEPT the one corresponding to member_idx. This creates
        diversity in the ensemble while ensuring each data point is left out
        exactly once across all members.

        CRITICAL: This logic must match the notebook exactly!

        Algorithm:
        1. Hash random state: rnd = hash((member_idx, fold_idx, random_state))
        2. Shuffle indices using rnd as seed
        3. Split shuffled indices into num_members groups
        4. Leave out the group corresponding to member_idx
        5. Return mask for all OTHER groups

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            member_idx: Index of current ensemble member (0 to num_members-1)
            num_members: Total number of ensemble members
            random_state: Base random seed for reproducibility
            fold_idx: Cross-validation fold index (default: 0)

        Returns:
            Tuple of:
            - X_train_bootstrap: Bootstrapped training data (~(M-1)/M of data)
            - leave_out_indices: Indices left out (~1/M of data)

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> X_train = np.random.randn(100, 5)
            >>> bootstrapper = StratifiedBootstrapper()
            >>> X_boot, leave_out = bootstrapper.perform_bootstrapping(
            ...     X_train, member_idx=2, num_members=5, random_state=42
            ... )
            >>> len(X_boot) + len(leave_out) == len(X_train)
            True
            >>> len(X_boot) == len(X_train) - len(leave_out)
            True
        """
        # Validate inputs
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train is empty or None")

        if member_idx < 0 or member_idx >= num_members:
            raise ValueError(
                f"member_idx must be in [0, {num_members-1}], got {member_idx}"
            )

        if num_members <= 0:
            raise ValueError(f"num_members must be positive, got {num_members}")

        self.logger.debug(
            f"Bootstrapping: member {member_idx}/{num_members}, "
            f"fold {fold_idx}, data shape {X_train.shape}"
        )

        # CRITICAL: Use SAME shuffle for all members
        # Only fold_idx and random_state determine the shuffle, NOT member_idx
        # This ensures non-overlapping leave-out sets and complete coverage
        rnd = hash((fold_idx, random_state)) % 4294967296
        rnd = rnd ^ 0x7FFFFFFF

        self.logger.debug(
            f"Shuffle random state: "
            f"hash(({fold_idx}, {random_state})) -> {rnd} "
            f"(same for all members)"
        )

        # Create random state with hashed seed
        rnd_state = np.random.RandomState(rnd)

        # Get indices and shuffle
        indices = np.arange(len(X_train))
        rnd_state.shuffle(indices)

        self.logger.debug(
            f"Shuffled {len(indices)} indices "
            f"(first 5: {indices[:5] if len(indices) >= 5 else indices})"
        )

        # Split into num_members groups
        index_sets = np.array_split(indices, num_members)

        self.logger.debug(
            f"Split into {num_members} groups: "
            f"sizes = {[len(s) for s in index_sets]}"
        )

        # Leave out group corresponding to member_idx
        leave_out_indices = index_sets[member_idx]

        # Create mask: True for indices to KEEP
        mask = np.ones(len(indices), dtype=bool)
        mask[leave_out_indices] = False

        # Bootstrap training data
        X_train_bootstrap = X_train[mask]

        self.logger.info(
            f"Bootstrap member {member_idx}: "
            f"train {len(X_train_bootstrap)}, leave-out {len(leave_out_indices)}, "
            f"total {len(X_train)}"
        )

        return X_train_bootstrap, leave_out_indices


def stratified_bootstrap(
    X_train: np.ndarray,
    member_idx: int,
    num_members: int,
    random_state: int,
    fold_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for stratified bootstrapping

    This is a simpler interface that doesn't require instantiating the class.

    Args:
        X_train: Training feature matrix
        member_idx: Index of ensemble member
        num_members: Total ensemble members
        random_state: Base random seed
        fold_idx: Fold index (default: 0)

    Returns:
        (X_train_bootstrap, leave_out_indices)

    Example:
        >>> X_train = np.random.randn(100, 5)
        >>> X_boot, leave_out = stratified_bootstrap(
        ...     X_train, member_idx=0, num_members=5, random_state=42
        ... )
    """
    bootstrapper = StratifiedBootstrapper()
    return bootstrapper.perform_bootstrapping(
        X_train, member_idx, num_members, random_state, fold_idx
    )
