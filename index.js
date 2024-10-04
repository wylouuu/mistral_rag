import MistralClient from "@mistralai/mistralai";
import { createClient } from "@supabase/supabase-js";

const mistralClient = new MistralClient(process.env.MISTRAL_API_KEY);
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

//1. Getting user input
const input = "what is the time i have to attend on the office?";

//2. Creating and embedding of the input
const embedding = await createEmbedding(input);

//3. Retrieving similar embedings / text chunks (aka "context")
const context = await retrieveMatches(embedding);

//4. Combine the input and the context in a prompt
//and using the chat API to generate a response
const response = await generateChatResponse(context, input);
console.log(response);

async function createEmbedding(input) {
  const embeddingResponse = await mistralClient.embeddings({
    model: "mistral-embed",
    input: [input],
  });
  return embeddingResponse.data[0].embedding;
}

async function retrieveMatches(embedding) {
  const { data } = await supabase.rpc("match_handbook_docs", {
    query_embedding: embedding, // Pass the embedding you want to compare
    match_threshold: 0.7, // Choose an appropriate threshold for your data
    match_count: 5, // Choose the number of matches
  });
  return data.map(chunk => chunk.content).join(" ");
}

async function generateChatResponse(context, query){
    const response = await mistralClient.chat({
        model: "mistral-large-latest",
        messages: [
            {
                role: 'system',
                content: 'you are the commander of the space flight and you talking to the user like an astronot that you respect. you also have tell them using some space reference on it'
            },
            {
                role: 'user',
                content: `Handbook context: ${context} - Question ${query}`
            }
        ]
    });
    return response.choices[0].message.content;
}
