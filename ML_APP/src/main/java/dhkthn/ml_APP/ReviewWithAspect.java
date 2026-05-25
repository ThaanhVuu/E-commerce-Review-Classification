package dhkthn.ml_APP;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "data_with_aspect")
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class ReviewWithAspect extends CustomerReview {
    @Column(columnDefinition = "TEXT")
    private String aspect;

    public ReviewWithAspect(String sentence, int rating, Integer label, String aspect) {
        super();
        super.setLabel(label);
        super.setRating(rating);
        super.setSentence(sentence);
        this.aspect = aspect;
    }

    public void setAspect(String aspect){
        if(aspect.isBlank()){
            throw new IllegalArgumentException("Aspect không được trống");
        }
        this.aspect = aspect;
    }
}
