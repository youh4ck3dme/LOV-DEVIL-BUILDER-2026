// @vitest-environment node
import { describe, expect, it, vi } from 'vitest';
import { configuredProviders, configuredProvidersWithMistralCatalog } from '../../../server/providers/runtime';
import { createApp } from '../../../server/index';

const env = { MISTRAL_API_KEY: 'test-only', MISTRAL_MODEL: 'mistral-large-latest' };
const card = (id: string, chat = true, aliases: string[] = []) => ({
  id, capabilities: { completion_chat: chat }, max_context_length: 262144, aliases,
});
const agentId = 'ag:01234567-89ab-cdef-0123-456789abcdef';

describe('verified Mistral model picker', () => {
  it('lists key-visible options in Hono and routes the chosen model upstream', async () => {
    const fetcher = vi.fn<typeof fetch>(async (_url, init) =>
      init?.method === 'POST'
        ? Response.json({ model: 'codestral-latest', choices: [{ message: { content: 'OK' } }] })
        : Response.json({ data: [
            card('codestral-2508', true, ['codestral-latest']),
            card('mistral-small-latest'),
            card('mistral-large-latest'),
            card('unlisted-model'),
          ] }),
    );
    const providers = await configuredProvidersWithMistralCatalog(env, fetcher);
    const app = createApp({}, providers.defaultProvider, { providers });
    const response = await app.request('/api/ai-models');
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      defaultId: 'mistral',
      models: [
        { id: 'mistral', model: 'mistral-large-latest', fallbackIds: [] },
        { id: 'codestral', model: 'codestral-latest', fallbackIds: [] },
        { id: 'mistral-small', model: 'mistral-small-latest', fallbackIds: [] },
      ],
    });
    expect(fetcher.mock.calls[0]).toEqual([
      'https://api.mistral.ai/v1/models',
      expect.objectContaining({ headers: { Authorization: 'Bearer test-only' }, redirect: 'error' }),
    ]);
    await providers.route('codestral')[0].provider.complete([{ role: 'user', content: 'Fixture' }]);
    expect(String(fetcher.mock.calls[1][0])).toBe('https://api.mistral.ai/v1/chat/completions');
    expect(JSON.parse(String(fetcher.mock.calls[1][1]?.body)).model).toBe('codestral-latest');
  });

  it('omits missing, archived, non-chat and duplicate alias options', async () => {
    for (const small of [
      card('mistral-small-2603', false),
      { ...card('mistral-small-2603'), archived: true },
      card('another-model'),
    ]) {
      const fetcher = vi.fn<typeof fetch>(async () => Response.json({
        data: [small, card('mistral-medium-2604', true, ['mistral-medium-latest'])],
      }));
      const providers = await configuredProvidersWithMistralCatalog(
        { ...env, MISTRAL_MODEL: 'mistral-medium-latest' }, fetcher,
      );
      expect(providers.catalog().models.map(model => model.id)).toEqual(['mistral']);
    }
  });

  it('keeps configured options when catalog lookup fails and skips custom endpoints', async () => {
    const rejected = vi.fn<typeof fetch>(async () => Response.json({ error: 'denied' }, { status: 401 }));
    expect((await configuredProvidersWithMistralCatalog(env, rejected)).catalog().models).toHaveLength(1);
    const fetcher = vi.fn<typeof fetch>();
    expect((await configuredProvidersWithMistralCatalog(
      { ...env, MISTRAL_BASE_URL: 'https://example.test/v1' }, fetcher,
    )).catalog().models).toHaveLength(1);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('respects the existing three-slot catalog and explicit defaults', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => Response.json({
      data: [card('mistral-small-latest'), card('codestral-2508', true, ['codestral-latest'])],
    }));
    const providers = await configuredProvidersWithMistralCatalog({
      ...env, AI_OPENAI_ENABLED: '1', OPENAI_API_KEY: 'openai-test', OPENAI_MODEL: 'openai-test-model',
      AI_DEFAULT_MODEL: 'openai',
    }, fetcher);
    expect(providers.catalog().defaultId).toBe('openai');
    expect(providers.catalog().models.map(model => model.id)).toEqual(['openai', 'mistral', 'codestral']);
  });
});

describe('Mistral Agent path', () => {
  it('calls agents completions with agent_id and keeps extra models on chat completions', async () => {
    const fetcher = vi.fn<typeof fetch>(async (url, init) => {
      if (init?.method !== 'POST') {
        return Response.json({ data: [
          card('codestral-2508', true, ['codestral-latest']),
          card('mistral-large-latest'),
          card('mistral-small-latest'),
          card('unlisted-model'),
        ] });
      }
      return Response.json({
        model: String(url).includes('/agents/') ? 'mistral-large-latest' : 'codestral-latest',
        choices: [{ message: { content: 'OK' } }],
        usage: { prompt_tokens: 3, completion_tokens: 1 },
      });
    });
    const providers = await configuredProvidersWithMistralCatalog({
      ...env, MISTRAL_AGENT_ID: agentId,
    }, fetcher);
    expect(providers.catalog().models.map(model => ({ id: model.id, model: model.model }))).toEqual([
      { id: 'mistral', model: agentId },
      { id: 'codestral', model: 'codestral-latest' },
      { id: 'mistral-large', model: 'mistral-large-latest' },
    ]);

    await providers.route('mistral')[0].provider.complete([{ role: 'user', content: 'Fixture' }]);
    const agentCall = fetcher.mock.calls.find(call => String(call[0]).endsWith('/agents/completions'));
    expect(String(agentCall?.[0])).toBe('https://api.mistral.ai/v1/agents/completions');
    const agentBody = JSON.parse(String(agentCall?.[1]?.body)) as { agent_id?: string; model?: string };
    expect(agentBody.agent_id).toBe(agentId);
    expect(agentBody).not.toHaveProperty('model');

    await providers.route('codestral')[0].provider.complete([{ role: 'user', content: 'Fixture' }]);
    const chatCalls = fetcher.mock.calls.filter(call => String(call[0]).endsWith('/chat/completions'));
    expect(chatCalls).toHaveLength(1);
    expect(String(chatCalls[0][0])).toBe('https://api.mistral.ai/v1/chat/completions');
    const chatBody = JSON.parse(String(chatCalls[0][1]?.body)) as { agent_id?: string; model?: string };
    expect(chatBody.model).toBe('codestral-latest');
    expect(chatBody).not.toHaveProperty('agent_id');
    expect(fetcher.mock.calls.filter(call => String(call[0]).includes('/agents/completions'))).toHaveLength(1);
  });

  it('rejects an agent id that is not shaped ag:… before any request', () => {
    const fetcher = vi.fn<typeof fetch>();
    for (const id of ['', 'agent-1', 'ag:short', 'ag:zzzzzzzz', 'AG:01234567-89ab-cdef-0123-456789abcdef']) {
      expect(() => configuredProviders({ ...env, MISTRAL_AGENT_ID: id }, fetcher)).toThrow('invalid_provider_configuration');
    }
    expect(fetcher).not.toHaveBeenCalled();
    expect(configuredProviders({ ...env, MISTRAL_AGENT_ID: agentId }, fetcher).catalog().models[0]).toMatchObject({
      id: 'mistral', model: agentId,
    });
    expect(fetcher).not.toHaveBeenCalled();
  });
});
