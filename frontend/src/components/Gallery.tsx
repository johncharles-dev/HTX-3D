import { useEffect, useState, useCallback } from 'react';
import { Trash2, Download, Eye, ChevronLeft, ChevronRight } from 'lucide-react';
import { getGallery, deleteGalleryItem } from '../api/client';
import type { GalleryItem, ExportFile } from '../types';

interface Props {
  onPreview: (exports: ExportFile[]) => void;
}

export default function Gallery({ onPreview }: Props) {
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const perPage = 12;

  const load = useCallback(async () => {
    try {
      const data = await getGallery(page, perPage);
      setItems(data.items);
      setTotal(data.total);
    } catch {
      // API not available
    }
  }, [page]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (taskId: string) => {
    if (!confirm('Delete this generation?')) return;
    await deleteGalleryItem(taskId);
    load();
  };

  const totalPages = Math.ceil(total / perPage);

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 rounded-2xl bg-bg-tertiary flex items-center justify-center">
            <Eye className="w-8 h-8 text-text-muted" />
          </div>
          <p className="text-sm text-text-muted">No generations yet</p>
          <p className="text-xs text-text-muted/60 mt-1">Your generated 3D models will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {items.map((item) => (
          <div
            key={item.task_id}
            className="bg-bg-secondary border border-border rounded-xl overflow-hidden group hover:border-border-hover transition-colors"
          >
            {/* Thumbnail */}
            <div className="aspect-square bg-bg-tertiary relative">
              {item.thumbnail_url ? (
                <img src={item.thumbnail_url} alt="" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-text-muted">
                  <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 3L2 9l10 6 10-6-10-6z" />
                    <path d="M2 17l10 6 10-6" />
                    <path d="M2 13l10 6 10-6" />
                  </svg>
                </div>
              )}
              {/* Hover overlay */}
              <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                <button
                  onClick={() => onPreview(item.exports)}
                  className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                  title="Preview"
                >
                  <Eye className="w-4 h-4 text-white" />
                </button>
                <button
                  onClick={() => handleDelete(item.task_id)}
                  className="p-2 rounded-lg bg-red-500/20 hover:bg-red-500/40 transition-colors"
                  title="Delete"
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                </button>
              </div>
            </div>

            {/* Info */}
            <div className="p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-text-muted">Seed: {item.seed}</span>
                <span className="text-xs text-text-muted">
                  {new Date(item.created_at).toLocaleDateString()}
                </span>
              </div>
              <div className="flex gap-1 flex-wrap">
                {item.exports.map((exp) => (
                  <a
                    key={exp.format}
                    href={exp.url}
                    download={exp.filename}
                    className="text-xs px-1.5 py-0.5 rounded bg-accent/10 text-accent hover:bg-accent/20 transition-colors flex items-center gap-1"
                  >
                    <Download className="w-3 h-3" />
                    {exp.format.toUpperCase()}
                  </a>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-6">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="p-2 rounded-lg border border-border text-text-muted hover:border-border-hover disabled:opacity-30"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-text-secondary">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="p-2 rounded-lg border border-border text-text-muted hover:border-border-hover disabled:opacity-30"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
