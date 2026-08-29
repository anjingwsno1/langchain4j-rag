package basic;


import Key.ApiKeys;
import Utils.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;

import static Utils.Utils.toPath;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocuments;

public class DocumentChatWithIngest {

    public static void main(String[] args) {

        //批量加载文档到内存
        List<Document> documents = loadDocuments(toPath("documents/"));

        //通过EmbeddingStore存储对象
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore();

        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .baseUrl(ApiKeys.BASE_URL)
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName("text-embedding-3-small") // deepseek用这个，如果是gpt就text-embedding-3-small
                .timeout(Duration.ofSeconds(60))
                .build();

        //创建一个EmbeddingStoreIngestor
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        ingestor.ingest(documents);

        //创建ContentRetriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3) // 可选：最多返回几条，默认3
                .minScore(0.0) // 可选：最低相似度
                .build();

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .baseUrl(ApiKeys.BASE_URL)
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        String userQuery = "我可以取消我的预订吗？";
        String answer = assistant.answer(userQuery);
        System.out.println(answer);
    }
}
