import React, { useState, useRef, useEffect, useId } from 'react';
import { ChevronDownIcon } from './Icons';

interface SettingControlProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  options: string[];
  onChange: (val: string) => void;
  hasChevron?: boolean;
}

export const SettingControl: React.FC<SettingControlProps> = ({
  icon,
  label,
  value,
  options,
  onChange,
  hasChevron = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const labelId = useId();

  // Close dropdown when clicking outside or pressing Escape
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const handleSelect = (option: string) => {
    onChange(option);
    setIsOpen(false);
    triggerRef.current?.focus();
  };

  return (
    <div className="setting-control-row" ref={containerRef}>
      {/* Left side: Icon and Label */}
      <div className="setting-label-group">
        <span className="setting-icon" aria-hidden="true">
          {icon}
        </span>
        <span id={labelId} className="setting-label-text">
          {label}
        </span>
      </div>

      {/* Right side: Functional dark pill control */}
      <div className="setting-control-wrapper">
        <button
          ref={triggerRef}
          type="button"
          className={`setting-pill-button ${isOpen ? 'active' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
          aria-haspopup="listbox"
          aria-expanded={isOpen}
          aria-labelledby={`${labelId} ${triggerRef.current?.id || ''}`}
          aria-controls={menuId}
        >
          <span className="setting-pill-value">{value}</span>
          {hasChevron && (
            <span className={`setting-pill-chevron ${isOpen ? 'rotate' : ''}`}>
              <ChevronDownIcon size={14} />
            </span>
          )}
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <ul
            id={menuId}
            role="listbox"
            aria-labelledby={labelId}
            className="setting-dropdown-menu"
          >
            {options.map((opt) => {
              const isSelected = opt === value;
              return (
                <li
                  key={opt}
                  role="option"
                  aria-selected={isSelected}
                  className={`setting-dropdown-item ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleSelect(opt)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSelect(opt);
                    }
                  }}
                  tabIndex={0}
                >
                  <span className="dropdown-item-label">{opt}</span>
                  {isSelected && <span className="dropdown-check" aria-hidden="true">✓</span>}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};
