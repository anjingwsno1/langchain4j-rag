package basic;


import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class EmbeddingStoreSearch {

    public static void main(String[] args) {

        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        TextSegment segment1 = TextSegment.from("我喜欢乒乓球");
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);

        TextSegment segment2 = TextSegment.from("今天天气不错");
        Embedding embedding2 = embeddingModel.embed(segment2).content();
        embeddingStore.add(embedding2, segment2);

        // 查询嵌入
        Embedding queryEmbedding = embeddingModel.embed("你最喜欢的运动是什么？").content();
        EmbeddingSearchRequest request = new EmbeddingSearchRequest(queryEmbedding, 1, 0.0, null);

        //执行检索
        EmbeddingSearchResult<TextSegment> result = embeddingStore.search(request);
        List<EmbeddingMatch<TextSegment>> relevant = result.matches();
        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0); //获取第一个结果

        System.out.println(embeddingMatch.score());
        System.out.println(embeddingMatch.embedded().text());
    }
}
