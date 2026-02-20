import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X } from 'lucide-react';

interface Props {
  files: File[];
  onChange: (files: File[]) => void;
  multiple?: boolean;
  maxFiles?: number;
}

export default function ImageUpload({ files, onChange, multiple = false, maxFiles = 4 }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (multiple) {
        const combined = [...files, ...accepted].slice(0, maxFiles);
        onChange(combined);
      } else {
        onChange(accepted.slice(0, 1));
      }
    },
    [files, onChange, multiple, maxFiles],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
    multiple,
    maxFiles: multiple ? maxFiles : 1,
  });

  const removeFile = (index: number) => {
    onChange(files.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-accent bg-accent/5' : 'border-border hover:border-border-hover'}
          ${files.length > 0 ? 'py-4' : 'py-10'}`}
      >
        <input {...getInputProps()} />
        <Upload className="w-8 h-8 mx-auto mb-2 text-text-muted" />
        <p className="text-sm text-text-secondary">
          {isDragActive
            ? 'Drop image here...'
            : multiple
              ? `Drag & drop up to ${maxFiles} images, or click to browse`
              : 'Drag & drop an image, or click to browse'}
        </p>
        <p className="text-xs text-text-muted mt-1">PNG with transparent background recommended</p>
      </div>

      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, i) => (
            <div key={i} className="relative group w-20 h-20 rounded-lg overflow-hidden border border-border">
              <img
                src={URL.createObjectURL(file)}
                alt={file.name}
                className="w-full h-full object-cover"
              />
              <button
                onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                className="absolute top-0.5 right-0.5 bg-bg-primary/80 rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3 text-text-secondary" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
