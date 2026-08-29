package basic;

import Key.ApiKeys;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.output.Response;

public class EmbeddingObject {

    public static void main(String[] args) {

        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .baseUrl(ApiKeys.BASE_URL)
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName("text-embedding-3-small")
                .build();

        Response<Embedding> response = embeddingModel.embed("杭州有一个西湖");
        System.out.println(response);
    }
}
