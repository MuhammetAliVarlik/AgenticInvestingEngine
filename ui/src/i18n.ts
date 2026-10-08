import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import en from './locales/en.json';
import tr from './locales/tr.json';

export const LANGUAGES = [
  { code: 'en', label: 'EN', name: 'English' },
  { code: 'tr', label: 'TR', name: 'Türkçe' },
] as const;

// The interface language only: reports are written by the model in English.
void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: { en: { translation: en }, tr: { translation: tr } },
    fallbackLng: 'en',
    supportedLngs: ['en', 'tr'],
    load: 'languageOnly',
    interpolation: { escapeValue: false }, // React escapes output
    detection: {
      // A choice made in the app wins; otherwise the browser language decides.
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'ie_lang',
      caches: ['localStorage'],
    },
  });

const setDocumentLanguage = (lng: string) => {
  document.documentElement.lang = lng.startsWith('tr') ? 'tr' : 'en';
};
setDocumentLanguage(i18n.resolvedLanguage ?? 'en');
i18n.on('languageChanged', setDocumentLanguage);

export default i18n;
