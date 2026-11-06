"use strict";
const $ = (selector) => document.querySelector(selector);
const history = [];
let busy = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "服务暂不可用，请稍后重试。";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (response.status === 422) detail = "输入过长或格式不正确，请缩短后重试。";
    } catch { /* Keep the safe default error. */ }
    throw new Error(detail);
  }
  return response.json();
}

function scrollMessages() {
  const panel = $("#messages");
  panel.scrollTop = panel.scrollHeight;
}

function appendMessage(role, text, isError = false) {
  $("#welcome")?.remove();
  const item = element("article", `message ${role}${isError ? " error" : ""}`);
  const label = element("div", "message-label");
  label.append(element("strong", "", role === "user" ? "你" : "✦ 企业助手"));
  item.append(label, element("div", "message-text", text));
  $("#messages").append(item);
  scrollMessages();
  return item;
}

async function showDocument(id) {
  try {
    const doc = await request(`/api/documents/${encodeURIComponent(id)}`);
    $("#document-title").textContent = doc.title;
    $("#document-content").textContent = doc.content;
    $("#document-dialog").showModal();
  } catch (error) { $("#status").textContent = error.message; }
}

function renderResult(result) {
  const item = appendMessage("assistant", result.answer);
  if (result.tool_calls.length) {
    const trace = element("details", "trace");
    trace.append(element("summary", "", `已执行 ${result.tool_calls.length} 次工具调用 · 点击查看记录`));
    for (const call of result.tool_calls) {
      const label = call.name === "search_documents" ? "文档检索" : "订单查询";
      trace.append(element("p", "", `${label} · ${call.elapsed_ms} ms`));
      trace.append(element("pre", "", JSON.stringify({参数: call.arguments, 结果: call.result}, null, 2)));
    }
    item.append(trace);
  }
  if (result.sources.length) {
    const list = element("div", "source-list");
    for (const source of result.sources) {
      const button = element("button", "", `检索来源 · ${source.title} [${source.id}]`);
      button.type = "button";
      button.title = source.text;
      button.addEventListener("click", () => showDocument(source.document));
      list.append(button);
    }
    item.append(list);
  }
  item.append(element("div", "response-meta", `${result.mode === "demo" ? "离线规则演示" : "大模型工具调用"} · ${(result.elapsed_ms / 1000).toFixed(2)} s · ${result.request_id.slice(0, 8)}`));
  scrollMessages();
}

async function sendQuestion(question) {
  if (busy || !question.trim()) return;
  busy = true;
  $("#send").disabled = true;
  $("#clear").disabled = true;
  $("#question").value = "";
  $("#status").textContent = "正在处理问题并调用所需工具……";
  appendMessage("user", question);
  try {
    const result = await request("/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: question, history: history.slice(-12)}),
      signal: AbortSignal.timeout(125000),
    });
    renderResult(result);
    history.push({role: "user", content: question}, {role: "assistant", content: result.answer.slice(0, 4000)});
    if (history.length > 12) history.splice(0, history.length - 12);
    $("#status").textContent = "";
  } catch (error) {
    const message = error.name === "TimeoutError" ? "请求超时，请稍后重试。" : error.message;
    appendMessage("assistant", message, true);
    $("#status").textContent = "本次请求未完成，可重新输入问题。";
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#clear").disabled = false;
    $("#question").focus();
  }
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion($("#question").value.trim());
});
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.dataset.question));
});
$("#clear").addEventListener("click", () => location.reload());
$("#close-dialog").addEventListener("click", () => $("#document-dialog").close());

async function initialize() {
  try {
    const [config, docs] = await Promise.all([request("/api/config"), request("/api/documents")]);
    $("#mode-badge").textContent = config.mode === "demo" ? "● 离线演示模式" : `● 大模型模式 · ${config.model}`;
    $("#mode-note").textContent = config.mode === "demo"
      ? "当前为规则演示，工具真实执行；接入 API 后启用大模型决策。"
      : config.ready ? "大模型已配置 · 工具按需调用 · 回答请结合原文核验。" : "尚未配置服务端 API 密钥。";
    $("#doc-count").textContent = docs.length;
    for (const doc of docs) {
      const button = element("button", "", `▤  ${doc.title.replace("（演示）", "")}`);
      button.addEventListener("click", () => showDocument(doc.id));
      $("#documents").append(button);
    }
  } catch { $("#mode-note").textContent = "服务连接失败，请检查后端是否启动。"; }
}
initialize();
