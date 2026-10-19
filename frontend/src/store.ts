import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from './api';
import type { Conversation, DocumentInfo, Message, Theme, User } from './types';

type AuthStatus = 'loading' | 'authenticated' | 'anonymous';

interface AppState {
  // UI preference — kept locally for instant paint; synced to /settings when signed in.
  theme: Theme;
  setTheme: (theme: Theme) => void;

  authStatus: AuthStatus;
  user: User | null;
  initAuth: () => Promise<void>;
  setSession: (user: User) => void;
  setUser: (user: User) => void;
  logout: () => Promise<void>;
  sessionExpired: () => void;

  conversations: Conversation[];
  conversationsTotal: number;
  activeConversationId: string | null;
  messages: Record<string, Message[]>;
  messageTotals: Record<string, number>;
  documents: DocumentInfo[];
  documentsTotal: number;
  conversationDocuments: Record<string, string[]>;

  loadConversations: () => Promise<void>;
  loadMoreConversations: () => Promise<void>;
  loadMessages: (conversationId: string) => Promise<void>;
  loadEarlierMessages: (conversationId: string) => Promise<void>;
  bumpMessageTotal: (conversationId: string, by: number) => void;
  loadDocuments: () => Promise<void>;
  loadMoreDocuments: () => Promise<void>;
  setActiveConversation: (id: string | null) => void;
  upsertConversation: (conv: Conversation) => void;
  removeConversation: (id: string) => void;
  setMessages: (conversationId: string, messages: Message[]) => void;
  upsertMessage: (conversationId: string, message: Message) => void;
  setDocuments: (docs: DocumentInfo[]) => void;
  upsertDocument: (doc: DocumentInfo) => void;
  removeDocument: (id: string) => void;
  setConversationDocuments: (conversationId: string, ids: string[]) => void;
  resetData: () => void;
}

const emptyData = {
  conversations: [] as Conversation[],
  conversationsTotal: 0,
  activeConversationId: null as string | null,
  messages: {} as Record<string, Message[]>,
  messageTotals: {} as Record<string, number>,
  documents: [] as DocumentInfo[],
  documentsTotal: 0,
  conversationDocuments: {} as Record<string, string[]>,
};

const byUpdatedDesc = (a: Conversation, b: Conversation) => b.updated_at.localeCompare(a.updated_at);

function mergeById<T extends { id: string }>(first: T[], second: T[]): T[] {
  const seen = new Set(first.map((x) => x.id));
  return [...first, ...second.filter((x) => !seen.has(x.id))];
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      setTheme: (theme) => set({ theme }),

      authStatus: 'loading',
      user: null,

      /** App start: no token in memory, so try the HttpOnly refresh cookie. */
      initAuth: async () => {
        // Never signed in on this browser: no cookie to try, so skip the request (it would only be a 401).
        if (!api.mayHaveSession()) {
          set({ authStatus: 'anonymous', user: null });
          return;
        }
        const session = await api.refresh();
        if (session) set({ authStatus: 'authenticated', user: session.user });
        else set({ authStatus: 'anonymous', user: null });
      },
      setSession: (user) => set({ authStatus: 'authenticated', user }),
      setUser: (user) => set({ user }),
      logout: async () => {
        await api.logout();
        set({ authStatus: 'anonymous', user: null, ...emptyData });
      },
      sessionExpired: () => {
        if (get().authStatus === 'authenticated') set({ authStatus: 'anonymous', user: null, ...emptyData });
      },

      ...emptyData,

      loadConversations: async () => {
        // Refresh the first page; keep already-loaded later pages.
        const page = await api.listConversations();
        set((s) => ({
          conversations: mergeById(page.items, s.conversations).sort(byUpdatedDesc),
          conversationsTotal: page.total,
        }));
      },
      loadMoreConversations: async () => {
        const page = await api.listConversations(30, get().conversations.length);
        set((s) => ({ conversations: mergeById(s.conversations, page.items), conversationsTotal: page.total }));
      },
      loadMessages: async (conversationId) => {
        const page = await api.listMessages(conversationId);
        set((s) => ({
          messages: { ...s.messages, [conversationId]: [...page.items].reverse() },
          messageTotals: { ...s.messageTotals, [conversationId]: page.total },
        }));
      },
      loadEarlierMessages: async (conversationId) => {
        const loaded = get().messages[conversationId] ?? [];
        const page = await api.listMessages(conversationId, 50, loaded.length);
        const earlier = [...page.items].reverse().filter((m) => !loaded.some((x) => x.id === m.id));
        set((s) => ({
          messages: { ...s.messages, [conversationId]: [...earlier, ...(s.messages[conversationId] ?? [])] },
          messageTotals: { ...s.messageTotals, [conversationId]: page.total },
        }));
      },
      bumpMessageTotal: (conversationId, by) =>
        set((s) => ({ messageTotals: { ...s.messageTotals, [conversationId]: (s.messageTotals[conversationId] ?? 0) + by } })),
      loadDocuments: async () => {
        const page = await api.listDocuments(50, 0);
        set((s) => ({ documents: mergeById(page.items, s.documents.slice(page.items.length)), documentsTotal: page.total }));
      },
      loadMoreDocuments: async () => {
        const page = await api.listDocuments(50, get().documents.length);
        set((s) => ({ documents: mergeById(s.documents, page.items), documentsTotal: page.total }));
      },
      setActiveConversation: (id) => set({ activeConversationId: id }),
      upsertConversation: (conv) =>
        set((s) => {
          const rest = s.conversations.filter((c) => c.id !== conv.id);
          const isNew = rest.length === s.conversations.length;
          return { conversations: [conv, ...rest].sort(byUpdatedDesc), conversationsTotal: s.conversationsTotal + (isNew ? 1 : 0) };
        }),
      removeConversation: (id) =>
        set((s) => {
          const { [id]: _removed, ...messages } = s.messages;
          return {
            conversations: s.conversations.filter((c) => c.id !== id),
            conversationsTotal: Math.max(0, s.conversationsTotal - 1),
            messages,
            activeConversationId: s.activeConversationId === id ? null : s.activeConversationId,
          };
        }),
      setMessages: (conversationId, messages) => set((s) => ({ messages: { ...s.messages, [conversationId]: messages } })),
      upsertMessage: (conversationId, message) =>
        set((s) => {
          const list = s.messages[conversationId] ?? [];
          const idx = list.findIndex((m) => m.id === message.id);
          const next = idx === -1 ? [...list, message] : list.map((m, i) => (i === idx ? message : m));
          return { messages: { ...s.messages, [conversationId]: next } };
        }),
      setDocuments: (documents) => set({ documents }),
      upsertDocument: (doc) =>
        set((s) => {
          const exists = s.documents.some((d) => d.id === doc.id);
          return {
            documents: exists ? s.documents.map((d) => (d.id === doc.id ? doc : d)) : [doc, ...s.documents],
            documentsTotal: s.documentsTotal + (exists ? 0 : 1),
          };
        }),
      removeDocument: (id) =>
        set((s) => ({
          documents: s.documents.filter((d) => d.id !== id),
          documentsTotal: Math.max(0, s.documentsTotal - 1),
          conversationDocuments: Object.fromEntries(
            Object.entries(s.conversationDocuments).map(([k, ids]) => [k, ids.filter((d) => d !== id)]),
          ),
        })),
      setConversationDocuments: (conversationId, ids) =>
        set((s) => ({ conversationDocuments: { ...s.conversationDocuments, [conversationId]: ids } })),
      resetData: () => set({ ...emptyData }),
    }),
    {
      name: 'advancedrag-ui',
      // Only the theme is persisted. No tokens, user data or chats in localStorage.
      partialize: (s) => ({ theme: s.theme }),
    },
  ),
);

api.onSessionExpired = () => useAppStore.getState().sessionExpired();
