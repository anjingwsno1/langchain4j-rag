package basic;

import Key.ApiKeys;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static Utils.Utils.toPath;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class DocumentChat {

    public static void main(String[] args) {

        //通过FileSystemDocumentLoader加载本地文件到内存中
        Document document = loadDocument(toPath("documents/sushi.txt"), new TextDocumentParser());

        //通过DocumentSplitter对文档进行分割
        DocumentSplitter splitter = DocumentSplitters.recursive(100, 0, new OpenAiTokenizer("gpt-4o-mini"));
        List<TextSegment> segments = splitter.split(document);

        //使用嵌入模型转换嵌入对象
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        //通过EmbeddingStore存储对象
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore();
        embeddingStore.addAll(embeddings, segments);

        String question = "苏东坡主要有哪些代表作品？";
        Embedding queryEmbedding = embeddingModel.embed(question).content();

        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 3, 0.7);
        String information = relevant.stream().map(match -> match.embedded().text()).collect(Collectors.joining("\n\n"));



        //构建提示词模板，动态创建提示词
        PromptTemplate promptTemplate = PromptTemplate.from(
                "尽力回答以下问题:\n"
                        + "\n"
                        + "问题:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "请根据以下信息回答:\n"
                        + "{{information}}"
        );

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", information);

        Prompt prompt = promptTemplate.apply(variables);

        //发送提示词给到大模型进行推演
        ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
                .baseUrl(ApiKeys.BASE_URL)
                .apiKey(ApiKeys.OPENAI_API_KEY)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build();
        AiMessage aiMessage = chatLanguageModel.generate(prompt.toUserMessage()).content();

        String answer = aiMessage.text();
        System.out.println(answer);
    }
}
