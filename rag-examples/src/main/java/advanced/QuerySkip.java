package advanced;

import Key.ApiKeys;
import Utils.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;

import static Utils.Utils.startConversationWith;
import static Utils.Utils.toPath;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;

public class QuerySkip {
    public static void main(String[] args) {

        Assistant assistant = createAssistant();

        // 首先说：“今天天气怎么样？”
        // 然后说：“我能取消预约吗？”
        startConversationWith(assistant);
    }

    private static Assistant createAssistant() {

        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .baseUrl(ApiKeys.BASE_URL)
                .modelName("gpt-4o-mini")
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .build();

        //创建一个用于实现查询跳过的QueryRouter
        QueryRouter queryRouter = new QueryRouter() {

            private final PromptTemplate promptTemplate = PromptTemplate.from(
                "以下查询是否与汽车租赁公司的业务相关？" +
                        "只回答'是'、'否'或者'可能'" +
                        "查询：{{it}}"
            );

            @Override
            public Collection<ContentRetriever> route(Query query) {
                Prompt prompt = promptTemplate.apply(query.text());

                AiMessage aiMessage = chatLanguageModel.generate(prompt.toUserMessage()).content();
                System.out.println("LLM的输出：" + aiMessage.text());

                if(aiMessage.text().toLowerCase().contains("否")) {
                    return emptyList();
                }

                return singletonList(contentRetriever);
            }
        };

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        return AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    private static EmbeddingStore<TextSegment> embed(Path documentPath, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }
}
