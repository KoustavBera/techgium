import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import TranslateRoundedIcon from '@mui/icons-material/TranslateRounded';
import KeyboardArrowDownRoundedIcon from '@mui/icons-material/KeyboardArrowDownRounded';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिंदी' },
  { code: 'bn', label: 'বাংলা' },
  { code: 'or', label: 'ଓଡ଼ିଆ' },
];

export default function LanguageSwitcher() {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsOpen(false);
  };

  const currentLang = LANGUAGES.find((l) => l.code === i18n.language) || LANGUAGES[0];

  return (
    <div className="relative w-full" ref={dropdownRef}>
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full px-4 py-2.5 rounded-xl text-sm font-medium
                   bg-white/60 backdrop-blur-md border border-[rgba(196,199,197,0.4)]
                   text-[var(--md-on-surface-variant)] shadow-sm touch-target"
        style={{ fontFamily: "'Roboto Flex', sans-serif" }}
        whileHover={{ scale: 1.01, backgroundColor: 'rgba(255, 255, 255, 0)' }}
        whileTap={{ scale: 0.98 }}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2.5">
          <TranslateRoundedIcon style={{ fontSize: 18, color: 'var(--md-primary)' }} />
          <span>{currentLang.label}</span>
        </div>
        <motion.div animate={{ rotate: isOpen ? 180 : 0 }}>
          <KeyboardArrowDownRoundedIcon style={{ fontSize: 18 }} />
        </motion.div>
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
            className="absolute bottom-full left-0 right-0 mb-2 p-1.5 rounded-xl
                       bg-white/80 backdrop-blur-xl border border-[rgba(255, 255, 255, 0.4)] shadow-lg z-50"
          >
            {LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                onClick={() => changeLanguage(lang.code)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors
                           ${i18n.language === lang.code
                    ? 'bg-[var(--md-primary-container)] text-[var(--md-on-primary-container)] font-semibold'
                    : 'text-[var(--md-on-surface)] hover:bg-[var(--md-surface-variant)]'
                  }`}
                style={{ fontFamily: "'Roboto Flex', sans-serif" }}
              >
                {lang.label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
