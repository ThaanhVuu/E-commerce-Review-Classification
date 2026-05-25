package dhkthn.ml_APP;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Slf4j
public class Controller {

    private final Repository rp;
    private final ReviewWithAspectRepository reviewWithAspectRepository;

    @GetMapping("/getRawData")
    @Transactional
    public List<CustomerReview> getData(
            @RequestParam(defaultValue = "10") int size
    ) {
        Instant claimTime = Instant.now();
        int claimed = rp.claimBatch(claimTime, size);
        if (claimed == 0) return List.of();
        return rp.findClaimedBatch(claimTime);
    }

    @PatchMapping("/update")
    @Transactional
    public void updateOrigin(@RequestBody Payload payload) {
        CustomerReview origin = rp.findOne(
                (root, query, cb) -> cb.and(
                        cb.equal(root.get("id"), payload.id()),
                        cb.isNull(root.get("label"))
                )
        ).orElseThrow(() -> new RuntimeException("Không tìm thấy data này! Có thể người khác đã đánh label"));

        origin.setLabel(payload.originLabel());
        origin.setClaimedAt(null); // giải phóng claim sau khi label xong
        rp.save(origin);

        updateReviewWithAspect(payload, origin);
    }

    private void updateReviewWithAspect(Payload payload, CustomerReview customerReview) {
        List<ReviewWithAspect> reviewWithAspects = new ArrayList<>();
        payload.aspectAndLabels().forEach(aspectAndLabel -> reviewWithAspects.add(
                new ReviewWithAspect(
                        payload.sentence(),
                        customerReview.getRating(),
                        aspectAndLabel.label(),
                        aspectAndLabel.aspect()
                )
        ));
        reviewWithAspectRepository.saveAll(reviewWithAspects);
    }

    @GetMapping("/labeled")
    public Page<ReviewWithAspect> getLabeledData(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "12") int size
    ) {
        // Có thể sort theo id giảm dần để xem cái mới nhất trước
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "id"));
        return reviewWithAspectRepository.findAll(pageable);
    }

    @GetMapping("/countUnlabeled")
    public long countUnlabeled() {
        return rp.countUnlabeledExact();
    }

    @GetMapping("/countLabeled")
    public long countLabeled() {
        return rp.countLabeledExact();
    }

    @Scheduled(fixedDelay = 5 * 60 * 1000)
    @Transactional
    public void releaseExpiredClaims() {
        Instant expiredBefore = Instant.now().minus(30, ChronoUnit.MINUTES);
        int released = rp.releaseExpiredClaims(expiredBefore);
        if (released > 0) log.info("Released {} expired claims", released);
    }

    @EventListener(ApplicationReadyEvent.class)
    @Transactional
    public void onStartup() {
        Instant expiredBefore = Instant.now().minus(30, ChronoUnit.MINUTES);
        int released = rp.releaseExpiredClaims(expiredBefore);
        log.info("Startup: released {} expired claims", released);
    }
}