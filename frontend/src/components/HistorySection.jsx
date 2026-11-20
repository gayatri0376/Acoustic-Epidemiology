import React, { useState, useMemo } from 'react';
import { 
  Search, 
  Filter, 
  ArrowUpDown, 
  Calendar, 
  FileAudio, 
  ChevronRight, 
  Activity, 
  Play, 
  Sparkles,
  Stethoscope,
  Clock,
  Inbox
} from 'lucide-react';
import AudioPlayer from './AudioPlayer';
import { API_BASE_URL } from '../services/api';

function formatDate(dateStr) {
  if (!dateStr) return '—';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return dateStr;
  }
}

export default function HistorySection({ 
  recordings = [], 
  loading = false, 
  onSelectRecording,
  onAnalyzeNew 
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState('ALL');
  const [sortOrder, setSortOrder] = useState('newest'); // 'newest' | 'oldest'
  const [activeAudioId, setActiveAudioId] = useState(null);

  // Filter and sort recordings
  const filteredRecordings = useMemo(() => {
    let list = [...recordings];

    // Search query filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(r => {
        const name = (r.original_filename || '').toLowerCase();
        const pid = (r.patient_pseudo_id || '').toLowerCase();
        const site = (r.anatomical_location || '').toLowerCase();
        const cls = (r.latest_prediction?.predicted_class || '').toLowerCase();
        return name.includes(q) || pid.includes(q) || site.includes(q) || cls.includes(q);
      });
    }

    // Category filter
    if (filterCategory !== 'ALL') {
      list = list.filter(r => {
        const cls = r.latest_prediction?.predicted_class;
        if (!cls) return filterCategory === 'UNANALYZED';
        if (filterCategory === 'NORMAL') return cls.toLowerCase() === 'normal';
        if (filterCategory === 'CAS') return cls.includes('CAS') && !cls.includes('&');
        if (filterCategory === 'DAS') return cls.includes('DAS') && !cls.includes('&');
        if (filterCategory === 'MIXED') return cls.includes('&');
        return true;
      });
    }

    // Sort order
    list.sort((a, b) => {
      const dateA = new Date(a.created_at || 0).getTime();
      const dateB = new Date(b.created_at || 0).getTime();
      return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
    });

    return list;
  }, [recordings, searchQuery, filterCategory, sortOrder]);

  return (
    <div className="history-experience-section" id="recordings-history">
      
      {/* Search & Filter Toolbar */}
      <div className="history-toolbar">
        
        {/* Search input */}
        <div className="search-box-wrap">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            placeholder="Search by filename, research ID, or anatomical location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="history-search-input"
          />
          {searchQuery && (
            <button 
              type="button" 
              onClick={() => setSearchQuery('')}
              className="clear-search-btn"
              aria-label="Clear search"
            >
              ×
            </button>
          )}
        </div>

        {/* Category Filter Pills */}
        <div className="category-filter-pills" role="radiogroup" aria-label="Filter by sound category">
          {[
            { id: 'ALL', label: 'All Audio' },
            { id: 'NORMAL', label: 'Normal' },
            { id: 'CAS', label: 'Wheezes (CAS)' },
            { id: 'DAS', label: 'Crackles (DAS)' },
            { id: 'MIXED', label: 'Mixed' },
          ].map(f => (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilterCategory(f.id)}
              className={`filter-pill ${filterCategory === f.id ? 'active' : ''}`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Sort select */}
        <div className="sort-wrap">
          <ArrowUpDown size={14} className="sort-icon" />
          <select 
            value={sortOrder} 
            onChange={(e) => setSortOrder(e.target.value)}
            className="history-sort-select"
            aria-label="Sort recordings by date"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>
        </div>

      </div>

      {/* Recordings Count Summary */}
      <div className="history-summary-strip">
        <span>Showing <strong>{filteredRecordings.length}</strong> of <strong>{recordings.length}</strong> respiratory recordings</span>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="history-loading-card">
          <div className="spinner-teal" />
          <p>Loading historical auscultation database records...</p>
        </div>
      )}

      {/* Empty State */}
      {!loading && filteredRecordings.length === 0 && (
        <div className="history-empty-card">
          <Inbox size={42} className="empty-icon" />
          <h3>No respiratory recordings found</h3>
          <p>
            {searchQuery || filterCategory !== 'ALL'
              ? 'Try adjusting your search query or category filters.'
              : 'There are no recordings in this repository yet. Upload your first pediatric audio file.'}
          </p>
          {onAnalyzeNew && (
            <button
              type="button"
              onClick={onAnalyzeNew}
              className="btn btn-primary"
              style={{ marginTop: '1rem' }}
            >
              <Sparkles size={16} /> Analyze New Recording
            </button>
          )}
        </div>
      )}

      {/* Chronological List / Cards */}
      {!loading && filteredRecordings.length > 0 && (
        <div className="history-cards-list">
          {filteredRecordings.map((rec) => {
            const hasPrediction = !!rec.latest_prediction;
            const predClass = rec.latest_prediction?.predicted_class;
            const audioSrc = rec.audio_file
              ? (rec.audio_file.startsWith('http') ? rec.audio_file : `${API_BASE_URL.replace(/\/+$/, '')}/${rec.audio_file.replace(/^\/+/, '')}`)
              : null;

            return (
              <div key={rec.id} className="history-record-card">
                
                <div className="record-card-main">
                  
                  {/* File Icon & Info */}
                  <div className="record-audio-meta">
                    <div className="record-icon-badge">
                      <FileAudio size={20} />
                    </div>
                    <div>
                      <h4 className="record-filename" title={rec.original_filename}>
                        {rec.original_filename || 'Auscultation Sample'}
                      </h4>
                      <div className="record-tags-row">
                        <span className="record-date font-mono">
                          <Clock size={12} /> {formatDate(rec.created_at)}
                        </span>
                        {rec.patient_pseudo_id && (
                          <span className="record-pill">
                            ID: {rec.patient_pseudo_id}
                          </span>
                        )}
                        {rec.anatomical_location && (
                          <span className="record-pill site-pill">
                            <Stethoscope size={11} /> {rec.anatomical_location.replace(/_/g, ' ')}
                          </span>
                        )}
                        {rec.patient_age_group && (
                          <span className="record-pill">
                            {rec.patient_age_group}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Prediction Pill & Action */}
                  <div className="record-action-group">
                    {hasPrediction ? (
                      <div className="record-prediction-badge">
                        <span className="pred-label">Acoustic Class:</span>
                        <span className={`pred-value-tag ${predClass === 'Normal' ? 'normal' : 'abnormal'}`}>
                          {predClass}
                        </span>
                      </div>
                    ) : (
                      <span className="badge badge-pending">Pending Analysis</span>
                    )}

                    <button
                      type="button"
                      onClick={() => onSelectRecording && onSelectRecording(rec)}
                      className="btn btn-secondary record-inspect-btn"
                    >
                      <span>Inspect Analysis</span>
                      <ChevronRight size={15} />
                    </button>
                  </div>

                </div>

                {/* Inline Audio Player preview toggle */}
                {audioSrc && (
                  <div className="record-card-player-wrap">
                    <AudioPlayer file={audioSrc} compact />
                  </div>
                )}

              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
