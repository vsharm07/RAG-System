class QueryCache:
    def __init__(self):
        self._cache = {}

    def _normalize(self, query):
        return " ".join(query.strip().lower().split())

    def _make_key(self, query, namespace=None):
        normalized_query = self._normalize(query)
        if namespace is None:
            return normalized_query

        return f"{namespace}::{normalized_query}"

    def get(self, query, namespace=None):
        return self._cache.get(self._make_key(query, namespace=namespace))

    def set(self, query, value, namespace=None):
        self._cache[self._make_key(query, namespace=namespace)] = value


query_cache = QueryCache()