package advanced;

import Key.ApiKeys;
import Utils.Assistant;
import com.healthmarketscience.jackcess.TableDefinition;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;


import java.util.function.Function;

import static dev.langchain4j.data.document.Metadata.metadata;
import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static org.assertj.core.api.Assertions.assertThat;


public class MetadataFiltering {

    ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
            .baseUrl(ApiKeys.BASE_URL)
            .apiKey(ApiKeys.OPENAI_API_KEY)
            .modelName("gpt-4o-mini")
            .logRequests(true)
            .build();

    EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

    @Test
    void Static_Metadata_Filter_Example() {

        // given
        TextSegment dogsSegment = TextSegment.from("Article about dogs ...", metadata("animal", "dog"));
        TextSegment birdsSegment = TextSegment.from("Article about birds ...", metadata("animal", "bird"));

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.add(embeddingModel.embed(dogsSegment).content(), dogsSegment);
        embeddingStore.add(embeddingModel.embed(birdsSegment).content(), birdsSegment);
        // embeddingStore contains segments about both dogs and birds

        Filter onlyDogs = metadataKey("animal").isEqualTo("dog");

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .filter(onlyDogs) // by specifying the static filter, we limit the search to segments only about dogs
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatLanguageModel)
                .contentRetriever(contentRetriever)
                .build();

        // when
        String answer = assistant.answer("Which animal?");

        // then
        assertThat(answer)
                .containsIgnoringCase("dog")
                .doesNotContainIgnoringCase("bird");
    }


    interface PersonalizedAssistant {

        String chat(@MemoryId String userId, @dev.langchain4j.service.UserMessage String userMessage);
    }

    @Test
    void Dynamic_Metadata_Filter_Example() {

        // given
        TextSegment user1Info = TextSegment.from("My favorite color is green", metadata("userId", "1"));
        TextSegment user2Info = TextSegment.from("My favorite color is red", metadata("userId", "2"));

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.add(embeddingModel.embed(user1Info).content(), user1Info);
        embeddingStore.add(embeddingModel.embed(user2Info).content(), user2Info);
        // embeddingStore contains information about both first and second user

        Function<Query, Filter> filterByUserId =
                (query) -> metadataKey("userId").isEqualTo(query.metadata().chatMemoryId().toString());

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                // by specifying the dynamic filter, we limit the search to segments that belong only to the current user
                .dynamicFilter(filterByUserId)
                .build();

        PersonalizedAssistant personalizedAssistant = AiServices.builder(PersonalizedAssistant.class)
                .chatLanguageModel(chatLanguageModel)
                .contentRetriever(contentRetriever)
                .build();

        // when
        String answer1 = personalizedAssistant.chat("1", "Which color would be best for a dress?");

        // then
        assertThat(answer1)
                .containsIgnoringCase("green")
                .doesNotContainIgnoringCase("red");

        // when
        String answer2 = personalizedAssistant.chat("2", "Which color would be best for a dress?");

        // then
        assertThat(answer2)
                .containsIgnoringCase("red")
                .doesNotContainIgnoringCase("green");
    }
}
