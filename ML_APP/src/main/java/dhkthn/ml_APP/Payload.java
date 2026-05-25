package dhkthn.ml_APP;

import java.util.List;

public record Payload(
        long id,
        List<AspectAndLabel> aspectAndLabels,
//        int rating,
        String sentence,
        int originLabel
) {
    public Payload {
        // Validate rating
//        if (rating < 1 || rating > 5) {
//            throw new IllegalArgumentException("Rating phải nằm trong khoảng 1 đến 5");
//        }
        // Validate originLabel
        if (!List.of(0, 1, 2).contains(originLabel)) {
            throw new IllegalArgumentException("originLabel phải là 0, 1, hoặc 2 (Negative, Neutral, Positive)");
        }
        // Validate aspectAndLabels
        if (aspectAndLabels == null || aspectAndLabels.isEmpty()) {
            throw new IllegalArgumentException("Aspect và label không được để trống");
        }
        for (AspectAndLabel item : aspectAndLabels) {
            if (item.aspect() == null || item.aspect().isBlank()) {
                throw new IllegalArgumentException("Aspect không được để trống");
            }
            if (!List.of(0, 1, 2).contains(item.label())) {
                throw new IllegalArgumentException("Label phải là 0, 1, hoặc 2 (Negative, Neutral, Positive)");
            }
        }
    }
}

record AspectAndLabel(
        int label,
        String aspect
) {}