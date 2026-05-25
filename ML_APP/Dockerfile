FROM bellsoft/liberica-openjre-alpine:25
WORKDIR /app
COPY build/libs/ML_APP-0.0.1-SNAPSHOT.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]