import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent, ReactNode } from 'react';
import {
  Copy, Edit2, FileText, LogOut, Menu, MoreHorizontal, Paperclip, RotateCcw,
  Search, Send, Settings, Share, Square, Trash2, UploadCloud, User as UserIcon, X,
} from 'lucide-react';
import { api, ApiError, errorMessage } from '../api';
import { useAppStore } from '../store';
import type { DocumentInfo, Message, StreamEvent, Theme } from '../types';
import {
  ACCEPTED_FILE_LABEL, ACCEPTED_FILE_TYPES, capitalize, dangerButtonStyle, displayName, ERROR_TEXT, formatBytes, MAX_UPLOAD_BYTES,
  formatRelative, initial, inputStyle, primaryButtonStyle,
} from './helpers';
import AssistantAnswer from './AssistantAnswer';
import { ConfirmDialog, FormMessage, Modal, StatusChip } from './ui';

type Streaming = { conversationId: string; phase: string; controller: AbortController } | null;
type Pending = { tempId: string; name: string; progress: number };
type PendingConfirm = { title: string; body: ReactNode; run: () => Promise<void> };

const NO_MESSAGES: Message[] = [];

const PHASES: Record<string, string> = {
  'agent.waking': 'Starting the AI service after a pause (this can take up to a minute)…',
  'agent.started': 'Thinking…',
  'retrieval.started': 'Searching your Knowledge Base…',
  'retrieval.completed': 'Reading sources…',
  'generation.started': 'Writing…',
};

export default function MainApp() {
  const store = useAppStore();
  const {
    user, theme, conversations, conversationsTotal, activeConversationId, messages, messageTotals, documents, documentsTotal,
  } = store;

  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [shareFor, setShareFor] = useState<string | null>(null);  // conversation being shared
  const [searchOpen, setSearchOpen] = useState(false);
  const [contextPanelOpen, setContextPanelOpen] = useState(false);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [pendingUploads, setPendingUploads] = useState<Pending[]>([]);
  const [streaming, setStreaming] = useState<Streaming>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [loadingMore, setLoadingMore] = useState<'chats' | 'messages' | 'documents' | null>(null);
  const [dragTarget, setDragTarget] = useState<'chat' | 'panel' | null>(null);
  const readyCount = documents.filter((d) => d.status === 'ready').length;

  const activeMessages = (activeConversationId && messages[activeConversationId]) || NO_MESSAGES;

  // ---------------------------------------------------------------- initial load
  useEffect(() => {
    store.loadConversations().catch((e) => setBanner(errorMessage(e)));
    store.loadDocuments().catch(() => {});
    api.getSettings().then((s) => store.setTheme(s.theme)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Messages for the open conversation. (Documents are not per chat: the Knowledge Base is searched in every chat.)
  useEffect(() => {
    if (!activeConversationId) return;
    const id = activeConversationId;
    if (!messages[id]) {
      setLoadingMessages(true);
      store.loadMessages(id).catch((e) => setBanner(errorMessage(e))).finally(() => setLoadingMessages(false));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversationId]);

  // Poll while any document is still being processed.
  const hasPending = documents.some((d) => d.status === 'uploaded' || d.status === 'processing');
  useEffect(() => {
    if (!hasPending) return;
    const timer = window.setInterval(() => {
      store.loadDocuments().catch(() => {});
    }, 2500);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasPending]);

  // Composer grows with its text up to the CSS max-height, then scrolls; shrinks again when cleared.
  const composerRef = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const max = parseFloat(getComputedStyle(el).maxHeight) || Infinity;
    el.style.height = `${Math.min(el.scrollHeight, max)}px`;
    el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden';
  }, [inputValue]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const newest = activeMessages[activeMessages.length - 1];
  const newestKey = newest ? `${newest.id}:${newest.content.length}:${newest.status}` : '';
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [newestKey]);

  const loadMore = async (what: 'chats' | 'messages' | 'documents') => {
    setLoadingMore(what);
    try {
      if (what === 'chats') await store.loadMoreConversations();
      else if (what === 'documents') await store.loadMoreDocuments();
      else if (activeConversationId) await store.loadEarlierMessages(activeConversationId);
    } catch (e) {
      setBanner(errorMessage(e));
    } finally {
      setLoadingMore(null);
    }
  };

  // ---------------------------------------------------------------- conversations
  const newChat = async () => {
    setSidebarOpen(window.innerWidth > 768);
    // Reuse an empty open chat instead of piling up blank conversations.
    if (activeConversationId && activeMessages.length === 0) return;
    try {
      const conv = await api.createConversation();
      store.upsertConversation(conv);
      store.setMessages(conv.id, []);
      store.setActiveConversation(conv.id);
    } catch (e) {
      setBanner(errorMessage(e));
    }
  };

  const openKnowledgeContext = () => {
    setContextPanelOpen(true);
    setSidebarOpen(true);
  };

  const closeSidebarOnMobile = () => {
    if (window.innerWidth <= 768) setSidebarOpen(false);
  };

  const openChat = (id: string) => {
    store.setActiveConversation(id);
    if (window.innerWidth <= 768) setSidebarOpen(false);
  };

  const commitRename = async (id: string) => {
    const title = editTitle.trim();
    setEditingChatId(null);
    if (!title) return;
    try {
      store.upsertConversation(await api.renameConversation(id, title));
    } catch (e) {
      setBanner(errorMessage(e));
    }
  };

  // Deletes ask for confirmation in an in-app dialog (never the browser's confirm()).
  const [pendingDelete, setPendingDelete] = useState<PendingConfirm | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const askToConfirm = (c: PendingConfirm) => {
    setDeleteError(null);
    setPendingDelete(c);
  };

  const runConfirmed = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await pendingDelete.run();
      setPendingDelete(null);
    } catch (e) {
      setDeleteError(errorMessage(e));
    } finally {
      setDeleting(false);
    }
  };
  const cancelConfirm = useCallback(() => setPendingDelete(null), []);

  const deleteChat = (id: string) => {
    const title = conversations.find((c) => c.id === id)?.title || 'this chat';
    askToConfirm({
      title: 'Delete chat?',
      body: <>This will delete <strong>{title}</strong>. This cannot be undone.</>,
      run: async () => {
        await api.deleteConversation(id);
        store.removeConversation(id);
      },
    });
  };

  // ---------------------------------------------------------------- streaming
  const consume = useCallback(
    async (conversationId: string, events: AsyncGenerator<StreamEvent>, controller: AbortController) => {
      let assistant: Message | null = null;
      let text = '';
      setStreaming({ conversationId, phase: 'Thinking…', controller });
      try {
        for await (const ev of events) {
          const data = ev.data as Record<string, unknown>;
          if (ev.event === 'message.created') {
            const userMsg = data.user_message as Message;
            assistant = data.assistant_message as Message;
            store.upsertMessage(conversationId, userMsg);
            store.upsertMessage(conversationId, assistant);
            store.bumpMessageTotal(conversationId, 2);
          } else if (ev.event in PHASES) {
            setStreaming((s) => (s ? { ...s, phase: PHASES[ev.event] } : s));
          } else if (ev.event === 'content.delta' && assistant) {
            text += String(data.delta ?? '');
            store.upsertMessage(conversationId, { ...assistant, content: text, status: 'streaming' });
          } else if (ev.event === 'content.replace' && assistant && typeof data.content === 'string') {
            // The verified answer replaces the streamed draft (unsupported claims removed, or the
            // "not in your Knowledge Base" answer); it is also what the server stored.
            text = data.content;
            store.upsertMessage(conversationId, { ...assistant, content: text, status: 'streaming' });
          } else if (ev.event === 'message.completed' && data.assistant_message) {
            store.upsertMessage(conversationId, data.assistant_message as Message);
          } else if (ev.event === 'error') {
            if (data.assistant_message) store.upsertMessage(conversationId, data.assistant_message as Message);
            else setBanner(String(data.message ?? 'The response could not be generated.'));
          }
        }
      } catch (e) {
        if (controller.signal.aborted) {
          // Server records the partial answer as cancelled; reflect that locally.
          if (assistant) store.upsertMessage(conversationId, { ...assistant, content: text, status: 'cancelled', error_code: 'CANCELLED' });
        } else {
          throw e;
        }
      } finally {
        setStreaming(null);
        store.loadConversations().catch(() => {});
      }
    },
    [store],
  );

  const handleSend = async () => {
    const content = inputValue.trim();
    if (!content || streaming) return;
    setBanner(null);
    let conversationId = activeConversationId;
    try {
      if (!conversationId) {
        const conv = await api.createConversation();
        store.upsertConversation(conv);
        store.setMessages(conv.id, []);
        store.setActiveConversation(conv.id);
        conversationId = conv.id;
      }
      setInputValue('');
      const controller = new AbortController();
      // No document ids: every ready document of the Knowledge Base is searched by the server.
      await consume(conversationId, api.streamMessage(conversationId, content, [], controller.signal), controller);
    } catch (e) {
      // Request rejected before the stream started: nothing was saved, give the text back.
      setInputValue(content);
      setBanner(errorMessage(e));
    }
  };

  const retry = async (message: Message) => {
    if (streaming || !activeConversationId) return;
    setBanner(null);
    const controller = new AbortController();
    try {
      store.upsertMessage(activeConversationId, { ...message, status: 'pending', content: '', error_code: null });
      await consume(activeConversationId, api.retryMessage(activeConversationId, message.id, controller.signal), controller);
    } catch (e) {
      store.upsertMessage(activeConversationId, message);
      setBanner(errorMessage(e));
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ---------------------------------------------------------------- documents (Knowledge Base)
  // The Knowledge Context panel is the only way to add documents. Every ready document is
  // searched in every chat; chats have no uploads or attachments of their own.
  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setBanner(null);
    const tooBig = Array.from(files).filter((f) => f.size > MAX_UPLOAD_BYTES);
    const queued = Array.from(files).filter((f) => f.size <= MAX_UPLOAD_BYTES).map((f) => ({ file: f, tempId: `${f.name}-${Math.random()}` }));
    // Oversized files are reported up front and never sent.
    const errors: string[] = tooBig.map((f) => `${f.name}: Files must be at most ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB.`);
    if (!queued.length) {
      setBanner(errors.join(' · '));
      return;
    }
    setPendingUploads((p) => [...p, ...queued.map(({ file, tempId }) => ({ tempId, name: file.name, progress: 0 }))]);
    await Promise.all(
      queued.map(async ({ file, tempId }) => {
        try {
          const doc = await api.uploadDocument(file, (progress) =>
            setPendingUploads((p) => p.map((x) => (x.tempId === tempId ? { ...x, progress } : x))),
          );
          store.upsertDocument(doc);
        } catch (e) {
          errors.push(`${file.name}: ${errorMessage(e)}`);
        } finally {
          setPendingUploads((p) => p.filter((x) => x.tempId !== tempId));
        }
      }),
    );
    if (errors.length) setBanner(errors.join(' · '));
  };

  const onFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    uploadFiles(e.target.files);
    e.target.value = '';
  };

  const dragProps = (target: 'chat' | 'panel') => ({
    onDragOver: (e: DragEvent) => {
      if (!e.dataTransfer.types.includes('Files')) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      if (dragTarget !== target) setDragTarget(target);
    },
    onDragLeave: (e: DragEvent) => {
      if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node | null)) setDragTarget(null);
    },
    onDrop: (e: DragEvent) => {
      if (!e.dataTransfer.files.length) return;
      e.preventDefault();
      setDragTarget(null);
      if (target === 'panel') {
        uploadFiles(e.dataTransfer.files);
        return;
      }
      // Dropped on the chat: nothing is uploaded from a chat. Point to the Knowledge Context panel.
      openKnowledgeContext();
      setBanner('Files are added in the Knowledge Context panel, not in a chat. Drop them there: once ready, they are searched in every chat.');
    },
  });

  const deleteDoc = (doc: DocumentInfo) => {
    askToConfirm({
      title: 'Delete document?',
      body: <>This will delete <strong>{doc.filename}</strong> from your Knowledge Base. It will no longer be searched in any chat.</>,
      run: async () => {
        await api.deleteDocument(doc.id);
        store.removeDocument(doc.id);
      },
    });
  };

  const reprocessDoc = async (doc: DocumentInfo) => {
    try {
      store.upsertDocument(await api.reprocessDocument(doc.id));
    } catch (e) {
      setBanner(errorMessage(e));
    }
  };

  const lastMessage = activeMessages[activeMessages.length - 1];
  const isStreamingHere = streaming?.conversationId === activeConversationId;
  const canSend = !!inputValue.trim() && !streaming;

  return (
    <div className="app-container">
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-hidden="true" />}
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h1>AdvancedRAG</h1>
          <div className="sidebar-header-actions">
            <button className="icon-btn" onClick={newChat} title="New chat"><Edit2 size={18} /></button>
            <button className="icon-btn" onClick={() => { closeSidebarOnMobile(); setSearchOpen(true); }} title="Search chats"><Search size={18} /></button>
            <button className="icon-btn" onClick={() => setContextPanelOpen(!contextPanelOpen)} title="Documents"><FileText size={18} /></button>
          </div>
        </div>

        {/* Knowledge Context: opens below the icons; the chats move down below it */}
        {contextPanelOpen && (
          <div className={`context-panel ${dragTarget === 'panel' ? 'drag-active' : ''}`} {...dragProps('panel')}>
            <div className="context-panel-header">
              <h3>Knowledge Context</h3>
              <button className="icon-btn" onClick={() => setContextPanelOpen(false)}><X size={18} /></button>
            </div>
            <div className="context-panel-body">
              <label className="upload-area" style={{ display: 'block' }}>
                <UploadCloud size={32} style={{ margin: '0 auto 12px' }} />
                <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: '4px' }}>
                  {dragTarget === 'panel' ? 'Drop to upload' : 'Upload Documents'}
                </div>
                <div style={{ fontSize: '0.8rem' }}>Click or drag files here · {ACCEPTED_FILE_LABEL}</div>
                <input name="documents" type="file" accept={ACCEPTED_FILE_TYPES} multiple style={{ display: 'none' }} onChange={onFileInput} data-testid="kb-upload" />
              </label>

              {(documents.length > 0 || pendingUploads.length > 0) && (
                <div style={{ marginTop: '24px' }}>
                  <h4 className="panel-heading">Your Knowledge Base</h4>
                  <p className="panel-note">Ready documents are searched automatically in every chat, new or old. Nothing to attach.</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {pendingUploads.map((p) => (
                      <div key={p.tempId} className="doc-row">
                        <div className="doc-row-main"><FileText size={16} color="var(--text-secondary)" /><span className="doc-name">{p.name}</span></div>
                        <StatusChip status="uploading" progress={p.progress} />
                      </div>
                    ))}
                    {documents.map((doc) => {
                      return (
                        <div key={doc.id} className="doc-row">
                          <div className="doc-row-main">
                            <FileText size={16} color="var(--text-secondary)" />
                            <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                          </div>
                          <div className="doc-row-meta">
                            <span>{formatBytes(doc.size_bytes)}</span>
                            <StatusChip status={doc.status} />
                            {doc.status === 'failed' && (
                              <button onClick={() => reprocessDoc(doc)} title="Retry processing"><RotateCcw size={14} /></button>
                            )}
                            <button onClick={() => deleteDoc(doc)} title="Delete"><Trash2 size={14} /></button>
                          </div>
                          {doc.processing_error && <div className="doc-error">{doc.processing_error}</div>}
                        </div>
                      );
                    })}
                    {documents.length < documentsTotal && (
                      <button className="load-more" onClick={() => loadMore('documents')} disabled={loadingMore === 'documents'}>
                        {loadingMore === 'documents' ? 'Loading…' : 'Load more documents'}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="chat-history">
          {conversations.length === 0 && (
            <div style={{ padding: '12px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No chats yet.</div>
          )}
          {conversations.map((chat) => (
            <div key={chat.id} className={`history-item ${chat.id === activeConversationId ? 'active' : ''}`} onClick={() => openChat(chat.id)}>
              {editingChatId === chat.id ? (
                <input
                  name="chat-title"
                  autoFocus
                  value={editTitle}
                  maxLength={200}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onBlur={() => commitRename(chat.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitRename(chat.id);
                    if (e.key === 'Escape') setEditingChatId(null);
                  }}
                  style={{ width: '100%', padding: '4px', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                />
              ) : (
                <>
                  <div className="history-item-title">{chat.title}</div>
                  <time className="history-item-time" dateTime={chat.updated_at} title={new Date(chat.updated_at).toLocaleString()}>
                    {formatRelative(chat.updated_at)}
                  </time>
                  <div className="history-item-actions">
                    <button title="Share" onClick={(e) => { e.stopPropagation(); closeSidebarOnMobile(); setShareFor(chat.id); }}><Share size={14} /></button>
                    <button title="Rename" onClick={(e) => { e.stopPropagation(); setEditingChatId(chat.id); setEditTitle(chat.title); }}><Edit2 size={14} /></button>
                    <button title="Delete" onClick={(e) => { e.stopPropagation(); deleteChat(chat.id); }}><Trash2 size={14} /></button>
                  </div>
                </>
              )}
            </div>
          ))}
          {conversations.length < conversationsTotal && (
            <button className="load-more" onClick={() => loadMore('chats')} disabled={loadingMore === 'chats'}>
              {loadingMore === 'chats' ? 'Loading…' : 'Load more chats'}
            </button>
          )}
        </div>

        <div className="sidebar-footer" onClick={() => setProfileMenuOpen(!profileMenuOpen)}>
          <div className="avatar">{initial(user)}</div>
          <span style={{ flex: 1, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName(user)}</span>
          <MoreHorizontal size={18} />
        </div>
      </div>

      {/* Profile Popup */}
      {profileMenuOpen && (
        <div className="popup-menu">
          <div className="popup-item" style={{ cursor: 'default', fontWeight: 600 }}>
            {displayName(user)} ({capitalize(user?.plan ?? 'free')})
          </div>
          <div className="popup-divider"></div>
          <div className="popup-item" onClick={() => { setProfileMenuOpen(false); setProfileOpen(true); }}>
            <UserIcon size={16} /> Profile
          </div>
          <div className="popup-item" onClick={() => { setProfileMenuOpen(false); setSettingsOpen(true); }}>
            <Settings size={16} /> Settings
          </div>
          <div className="popup-divider"></div>
          <div className="popup-item" onClick={() => { setProfileMenuOpen(false); store.logout(); }}>
            <LogOut size={16} /> Log out
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className={`main-chat ${dragTarget === 'chat' ? 'drag-active' : ''}`} {...dragProps('chat')}>
        {dragTarget === 'chat' && <div className="drop-overlay">Add documents in the Knowledge Context panel</div>}
        <div className="chat-header">
          <div className="header-left">
            <button className="icon-btn menu-toggle" onClick={() => setSidebarOpen(true)}><Menu size={20} /></button>
            <h2 style={{ opacity: 0 }} className="mobile-title">AdvancedRAG</h2>
          </div>
        </div>

        {banner && (
          <div className="banner" role="alert">
            <span>{banner}</span>
            <button className="icon-btn" onClick={() => setBanner(null)} aria-label="Dismiss"><X size={16} /></button>
          </div>
        )}

        <div className="messages-container">
          {loadingMessages && activeMessages.length === 0 ? (
            <div className="messages-loading" role="status">Loading conversation…</div>
          ) : activeMessages.length === 0 ? (
            <div style={{ margin: 'auto', textAlign: 'center' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '16px' }}>AdvancedRAG</h1>
              <p style={{ color: 'var(--text-secondary)' }}>How can I help you today?</p>
            </div>
          ) : (
            <>
            {activeConversationId && activeMessages.length < (messageTotals[activeConversationId] ?? 0) && (
              <button className="load-more" onClick={() => loadMore('messages')} disabled={loadingMore === 'messages'}>
                {loadingMore === 'messages' ? 'Loading…' : 'Load earlier messages'}
              </button>
            )}
            {activeMessages.map((msg) => (
              <div key={msg.id} className="message-wrapper">
                <div className={`message-avatar ${msg.role}`}>{msg.role === 'user' ? initial(user) : 'AG'}</div>
                <div className="message-content">
                  {msg.attachments.length > 0 && (
                    <div className="doc-list" style={{ marginBottom: '12px' }}>
                      {msg.attachments.map((doc) => (
                        <div key={doc.id} className="doc-chip"><Paperclip size={12} /> {doc.filename}</div>
                      ))}
                    </div>
                  )}
                  {msg.role === 'assistant'
                    ? <AssistantAnswer messageId={msg.id} content={msg.content} citations={msg.citations} />
                    : msg.content && <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>}
                  {msg.role === 'assistant' && (msg.status === 'pending' || msg.status === 'streaming') && isStreamingHere && (
                    <div className="stream-status">{streaming?.phase}</div>
                  )}
                  {msg.role === 'assistant' && (msg.status === 'failed' || msg.status === 'cancelled') && (
                    <div className="message-error">
                      <span>{ERROR_TEXT[msg.error_code ?? ''] ?? 'The response could not be generated.'}</span>
                      {msg.id === lastMessage?.id && !streaming && (
                        <button onClick={() => retry(msg)} className="retry-btn"><RotateCcw size={14} /> Retry</button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="composer-container">
          <div className="composer-wrapper">
            <div className="composer-input-row">
              <button className="icon-btn" style={{ padding: '6px', alignSelf: 'flex-start', marginTop: '2px' }}
                title="Open Knowledge Context" onClick={openKnowledgeContext}>
                <FileText size={20} />
              </button>
              <textarea
                id="composer"
                name="message"
                ref={composerRef}
                className="composer-textarea"
                placeholder="Ask AdvancedRAG anything..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
              />
              {streaming ? (
                <button className="icon-btn" onClick={() => streaming.controller.abort()} title="Stop generating"
                  style={{ backgroundColor: 'var(--btn-bg)', color: 'var(--btn-text)' }}>
                  <Square size={16} />
                </button>
              ) : (
                <button className="icon-btn" onClick={handleSend} disabled={!canSend} title="Send"
                  style={{ backgroundColor: canSend ? 'var(--btn-bg)' : 'transparent', color: canSend ? 'var(--btn-text)' : 'var(--text-secondary)' }}>
                  <Send size={18} />
                </button>
              )}
            </div>
          </div>
          <div className="composer-note">
            {readyCount > 0
              ? `Answers about your documents come only from your Knowledge Base (${readyCount} ready document${readyCount === 1 ? '' : 's'}), with citations.`
              : 'Add documents in the Knowledge Context panel to ask questions about them.'}
            {' '}AdvancedRAG can make mistakes. Consider verifying important information.
          </div>
        </div>
      </div>

      {searchOpen && <SearchModal onClose={() => setSearchOpen(false)} onOpen={(id) => { openChat(id); setSearchOpen(false); }} />}
      {profileOpen && <ProfileModal onClose={() => setProfileOpen(false)} />}
      {settingsOpen && <SettingsModal theme={theme} onClose={() => setSettingsOpen(false)} />}
      {pendingDelete && (
        <ConfirmDialog title={pendingDelete.title} confirmLabel="Delete" busy={deleting} error={deleteError}
          onConfirm={runConfirmed} onCancel={cancelConfirm}>
          {pendingDelete.body}
        </ConfirmDialog>
      )}
      {shareFor && (
        <ShareModal conversationId={shareFor === activeConversationId && !activeMessages.length ? null : shareFor} onClose={() => setShareFor(null)} />
      )}

    </div>
  );
}

// ---------------------------------------------------------------------------- modals

function SearchModal({ onClose, onOpen }: { onClose: () => void; onOpen: (id: string) => void }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<{ id: string; title: string; snippet?: string | null }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const q = query.trim();
    if (!q) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setLoading(true);
      setError(null);
      api.searchChats(q, controller.signal)
        .then((p) => setResults(p.items))
        .catch((e) => {
          if (!controller.signal.aborted) {
            setResults([]);
            setError(errorMessage(e));
          }
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ marginTop: '10vh', alignSelf: 'flex-start' }}>
        <div className="composer-wrapper" style={{ padding: '4px 12px', display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
          <Search size={18} style={{ color: 'var(--text-secondary)' }} />
          <input name="chat-search" autoFocus placeholder="Search chats..." value={query} maxLength={200} onChange={(e) => setQuery(e.target.value)}
            style={{ flex: 1, padding: '12px', fontSize: '1rem' }} />
        </div>
        {query.trim() && (
          <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '8px', fontWeight: 600 }}>
              {loading ? 'SEARCHING…' : error ? 'SEARCH FAILED' : results.length ? 'RESULTS' : 'NO RESULTS'}
            </span>
            {error && <div className="form-error" role="alert" style={{ paddingLeft: 8, marginTop: 0 }}>{error}</div>}
            {results.map((chat) => (
              <div key={chat.id} className="history-item" onClick={() => onOpen(chat.id)}
                style={{ padding: '12px', border: '1px solid var(--border-color)', flexDirection: 'column', alignItems: 'flex-start' }}>
                <div>{chat.title}</div>
                {chat.snippet && <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 4 }}>{chat.snippet}</div>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProfileModal({ onClose }: { onClose: () => void }) {
  const user = useAppStore((s) => s.user);
  const setUser = useAppStore((s) => s.setUser);
  const [name, setName] = useState(user?.full_name ?? '');
  const [msg, setMsg] = useState<{ error?: string; success?: string }>({});

  const saveName = async () => {
    try {
      setUser(await api.updateProfile(name));
      setMsg({ success: 'Profile saved.' });
    } catch (e) {
      setMsg({ error: errorMessage(e) });
    }
  };

  const fieldStyle = { ...inputStyle, padding: '8px', fontSize: '0.9rem', width: '220px' };

  return (
    <Modal title="Profile" onClose={onClose}>
      <div className="setting-row">
        <span style={{ fontWeight: 500 }}>Name</span>
        <input name="chat-name" value={name} maxLength={100} onChange={(e) => setName(e.target.value)} style={fieldStyle} />
      </div>
      <div className="setting-row" style={{ marginTop: '16px' }}>
        <span style={{ fontWeight: 500 }}>Email</span>
        <span style={{ color: 'var(--text-secondary)' }}>{user?.email}</span>
      </div>
      <div className="setting-row" style={{ marginTop: '16px' }}>
        <span style={{ fontWeight: 500 }}>Plan</span>
        <span style={{ color: 'var(--text-secondary)' }}>{capitalize(user?.plan ?? 'free')}</span>
      </div>
      <div className="setting-row" style={{ marginTop: '16px' }}>
        <span style={{ fontWeight: 500 }}>Member since</span>
        <span style={{ color: 'var(--text-secondary)' }}>{user ? new Date(user.created_at).toLocaleDateString() : ''}</span>
      </div>
      <FormMessage error={msg.error} success={msg.success} />
      <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={saveName} style={primaryButtonStyle}>Save</button>
      </div>
    </Modal>
  );
}

function SettingsModal({ theme, onClose }: { theme: Theme; onClose: () => void }) {
  const setTheme = useAppStore((s) => s.setTheme);
  const resetData = useAppStore((s) => s.resetData);
  const sessionExpired = useAppStore((s) => s.sessionExpired);
  const [error, setError] = useState<string | null>(null);
  const [asking, setAsking] = useState<'clear' | 'logout' | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const closeConfirm = useCallback(() => setAsking(null), []);

  const changeTheme = (value: Theme) => {
    setTheme(value);
    api.updateSettings({ theme: value }).catch((e) => setError(errorMessage(e)));
  };

  const clearAll = async () => {
    setBusy(true);
    setConfirmError(null);
    try {
      await api.clearAllData();
      resetData();
      setAsking(null);
      onClose();
    } catch (e) {
      setConfirmError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const logoutEverywhere = async () => {
    setBusy(true);
    setConfirmError(null);
    try {
      await api.logoutAll();
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 401)) {
        setConfirmError(errorMessage(e));
        setBusy(false);
        return;
      }
    }
    sessionExpired();
  };

  const ask = (what: 'clear' | 'logout') => {
    setConfirmError(null);
    setAsking(what);
  };

  return (
    <Modal title="Settings" onClose={onClose}>
      <div className="setting-row">
        <span style={{ fontWeight: 500 }}>Theme</span>
        <select value={theme} onChange={(e) => changeTheme(e.target.value as Theme)}>
          <option value="dark">Dark</option>
          <option value="light">Light</option>
          <option value="system">System</option>
        </select>
      </div>
      <FormMessage error={error} />
      <div className="setting-row" style={{ marginTop: '24px' }}>
        <button onClick={() => ask('logout')} style={{ ...primaryButtonStyle, width: '100%', padding: '10px' }}>Log out of all devices</button>
      </div>
      <div className="setting-row" style={{ marginTop: '12px' }}>
        <button onClick={() => ask('clear')} style={{ ...dangerButtonStyle, width: '100%', padding: '10px' }}>Clear All Data</button>
      </div>
      {asking === 'clear' && (
        <ConfirmDialog title="Clear all data?" confirmLabel="Delete everything" busy={busy} error={confirmError}
          onConfirm={clearAll} onCancel={closeConfirm}>
          This will delete <strong>all your chats and Knowledge Base documents</strong>. This cannot be undone.
        </ConfirmDialog>
      )}
      {asking === 'logout' && (
        <ConfirmDialog title="Log out of all devices?" confirmLabel="Log out" busyLabel="Logging out…" busy={busy}
          error={confirmError} onConfirm={logoutEverywhere} onCancel={closeConfirm}>
          You will be signed out on every device, including this one.
        </ConfirmDialog>
      )}
    </Modal>
  );
}

function ShareModal({ conversationId, onClose }: { conversationId: string | null; onClose: () => void }) {
  const [active, setActive] = useState<boolean | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!conversationId) return;
    api.getShareStatus(conversationId).then((s) => setActive(s.active)).catch((e) => setError(errorMessage(e)));
  }, [conversationId]);

  const createAndCopy = async () => {
    if (!conversationId) return;
    setBusy(true);
    setError(null);
    try {
      const share = await api.createShare(conversationId);
      setUrl(share.url);
      setActive(true);
      try {
        await navigator.clipboard.writeText(share.url);
        setCopied(true);
      } catch {
        setCopied(false); // clipboard blocked: the URL is shown for manual copy
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const revoke = async () => {
    if (!conversationId) return;
    try {
      await api.revokeShare(conversationId);
      setActive(false);
      setUrl(null);
      setCopied(false);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  return (
    <Modal title="Share link to chat" onClose={onClose}>
      {!conversationId ? (
        <p style={{ color: 'var(--text-secondary)' }}>Start a conversation to share it.</p>
      ) : (
        <>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Messages you send after creating your link won't be shared. Anyone with the URL will be able to view the shared chat.
          </p>
          {url && (
            <input name="share-link" readOnly value={url} onFocus={(e) => e.target.select()} style={{ ...inputStyle, fontSize: '0.85rem', marginBottom: 12 }} />
          )}
          {active && !url && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
              A link is already active. Creating a new link will stop the old one from working.
            </p>
          )}
          <FormMessage error={error} success={copied ? 'Link copied to clipboard.' : null} />
          <button className="new-chat-btn" disabled={busy} onClick={createAndCopy}
            style={{ width: '100%', margin: '12px 0 0', padding: '12px', backgroundColor: 'var(--btn-bg)', color: 'var(--btn-text)', fontWeight: 600, borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
            <Copy size={16} /> {active ? 'Create new link & copy' : 'Create link & copy'}
          </button>
          {active && (
            <button onClick={revoke} style={{ ...dangerButtonStyle, width: '100%', marginTop: 8, padding: '10px' }}>Revoke link</button>
          )}
        </>
      )}
    </Modal>
  );
}
