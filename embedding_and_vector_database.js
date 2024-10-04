import fs from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import MistralClient from "@mistralai/mistralai";
import { createClient } from "@supabase/supabase-js";

const mistralClient = new MistralClient(process.env.MISTRAL_API_KEY); 
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY);

async function splitDocument(path) {
  const response = await fs.readFileSync(path, "utf8");
  const text = await response.toString();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 40,
  });
  const output = await splitter.createDocuments([text]);
  const textArr = output.map((chunk) => chunk.pageContent);

  return textArr;
}

// console.log(await splitDocument('handbook.txt'));

// contoh dari embeddings

// const exampleChunk = 'We highly prioritize continuous learning so our team members can grow with new technologies beneficial to Lunexia.';

// const embeddingsResponse = await client.embeddings({
//     model:'mistral-embed',
//     input: [exampleChunk]
// })

// console.log(embeddingsResponse);

// contoh dari embeddings

const handbookChunks = await splitDocument("handbook.txt");

async function createEmbeddings(chunks) {
  const embeddings = await mistralClient.embeddings({
    model: "mistral-embed",
    input: chunks,
  });
  const data = chunks.map((chunk, i) => {
    return {
      content: chunk,
      embedding: embeddings.data[i].embedding,
    };
  });

  return data;
}

//insert data hasil embedding ke vector database
const data = await createEmbeddings(handbookChunks);
const response = await supabase.from("handbook_docs").insert(data);
console.log(response);