import { useState, useEffect } from 'react';
import type { Catalog } from '../types';

export function useCatalog() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/catalog')
      .then((r) => r.json())
      .then((data: Catalog) => setCatalog(data))
      .catch((err) => setError(err.message));
  }, []);

  return { catalog, error };
}
