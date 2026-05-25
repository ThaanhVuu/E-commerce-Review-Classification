package dhkthn.ml_APP;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;

public interface Repository extends JpaRepository<CustomerReview, Long>,
        JpaSpecificationExecutor<CustomerReview> {

    @Modifying
    @Query(value = """
        UPDATE data
        SET claimed_at = :now
        WHERE id IN (
            SELECT id FROM data
            WHERE label IS NULL
            AND claimed_at IS NULL
            ORDER BY id
            LIMIT :size
        )
    """, nativeQuery = true)
    int claimBatch(@Param("now") Instant now, @Param("size") int size);

    @Query("SELECT r FROM CustomerReview r WHERE r.claimedAt = :claimTime")
    List<CustomerReview> findClaimedBatch(@Param("claimTime") Instant claimTime);

    @Modifying
    @Query("""
        UPDATE CustomerReview r
        SET r.claimedAt = null
        WHERE r.claimedAt < :expiredBefore
        AND r.label IS NULL
    """)
    int releaseExpiredClaims(@Param("expiredBefore") Instant expiredBefore);

    @Query(value = "SELECT COUNT(*) FROM data WHERE label IS NOT NULL", nativeQuery = true)
    long countLabeledExact();

    @Query(value = "SELECT COUNT(*) FROM data WHERE label IS NULL", nativeQuery = true)
    long countUnlabeledExact();
}